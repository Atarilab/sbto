import numpy as np
import numpy.typing as npt
from typing import Any
from abc import ABC, abstractmethod
from scipy.stats import qmc, norm, beta
from scipy.special import betainc
from scipy.special import beta as beta_f
import numpy as np
from scipy.stats import norm
from scipy.special import gammaln
from scipy.optimize import root, minimize

Array = npt.NDArray[np.float64]

class SamplerAbstract(ABC):
    def __init__(self,
                 N_samples: int,
                 seed: int = 0,
                 quasi_random: bool = True,
                 **kwargs,
                 ):
        self.N_samples = N_samples
        self.quasi_random = quasi_random
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def sample(self, **params) -> Array:
        pass
    
    @abstractmethod
    def estimate_params(self, samples) -> Any:
        pass

class MultivariateNormal(SamplerAbstract):
    def sample(self, mean, cov, **kwargs):
        if not self.quasi_random:
            noise = self.rng.multivariate_normal(
                mean=mean,
                cov=cov,
                size=(self.N_samples,),
                check_valid="ignore",
                method="cholesky"
            )
        else:
            sampler = qmc.MultivariateNormalQMC(
                mean=mean,
                cov=cov,
                rng=self.rng,
                inv_transform=False,
            )
            noise = sampler.random(self.N_samples)

        return noise
    
    def estimate_params(self, samples):
        """
        samples [N, D]
        """
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        return mean, cov

class BetaMultivariateCopulas(SamplerAbstract):
    def __init__(
            self,
            N_samples: int,
            seed: int = 0,
            quasi_random: bool = True,
            **kwargs,
            ):
        super().__init__(N_samples, seed, quasi_random)
        self.normal_sampler = MultivariateNormal(
            N_samples,
            seed,
            quasi_random,
            )

    def sample(self, a, b, Sigma, **kwargs):
        """
        Multivariate Beta samples using Gaussian copula.
        From: https://twiecki.io/blog/2018/05/03/copulas/
        """
        mean = np.zeros_like(a)
        z = self.normal_sampler.sample(mean, Sigma)
        u = norm.cdf(z)
        b_multivariate = beta.ppf(u, a, b)
        return b_multivariate

    @staticmethod    
    def mode(a, b):
        return (a - 1.) / (a + b - 2.)

    @staticmethod 
    def delta_v(a, b, c):
        """
        How much the variance is reduced when applying mode regularization.
        """
        m = BetaMultivariateCopulas.mode(a, b)
        p = a*b
        s = a+b
        A = p / (s**2 * (s+1))
        B = (a + c*m) * (b + c * (1.-m))
        B /= (s + c)**2 * (s + c + 1.)
        return A - B

    def estimate_params(self, samples, eps=1e-6, c=1.):
        """
        Estimate Beta parameters (a, b) and Gaussian copula correlation Sigma
        using method of moments. Robust version ensuring positivity.
        samples: array of shape (N, D)
        """
        m = np.mean(samples, axis=0)
        v = np.var(samples, axis=0)
        # min_var = 1e-4
        # v[v < min_var] = min_var

        # Compute a, b
        temp = m * (1 - m) / v - 1.0
        a = m * temp
        b = (1 - m) * temp
        
        # a, b regularization
        # adding a delta_a delta_b such that:
        # mode(a, b) = mode(a + delta_a, b + delta_b)
        # delta_a + delta_b = c > 0
        # mode = BetaMultivariateCopulas.mode(a, b)
        # mode = np.clip(mode, eps, 1.0 - eps)
        # Since this reduces the variance,
        # we regularize only when empirical v is below the
        # delta_v
        # a += c * mode
        # b += c * (1. - mode)
        # delta_v = BetaMultivariateCopulas.delta_v(a, b, c)
        # a = np.where(v > 2*delta_v, a + c * mode, a)
        # b = np.where(v > 2*delta_v, b + c * (1. - mode), b)

        # Transform to copula space
        u = betainc(a, b, samples)
        u = np.clip(u, eps, 1 - eps)  # avoid infinities in norm.ppf

        z = norm.ppf(u)
        Sigma = np.corrcoef(z, rowvar=False)
        return a, b, Sigma


class KumaraswamyMultivariate(SamplerAbstract):
    def __init__(
            self,
            N_samples: int,
            seed: int = 0,
            quasi_random: bool = True,
            **kwargs,
            ):
        super().__init__(N_samples, seed, quasi_random)
        self.normal_sampler = MultivariateNormal(
            N_samples,
            seed,
            quasi_random,
            )
        self._a = None
        self._b = None
        
    @staticmethod
    def ppf(u, a, b):
        return (1 - (1 - u)**(1.0/b))**(1.0/a)

    @staticmethod
    def cdf(x, a, b):
        return 1 - (1 - x**a)**b

    def sample(self, a, b, Sigma, **kwargs):
        """
        Multivariate Kumaraswamy samples using Gaussian copula.
        """
        self._a, self._b = a, b
        mean = np.zeros_like(a)
        z = self.normal_sampler.sample(mean, Sigma)
        u = norm.cdf(z)               # transform to uniform
        x = self.ppf(u, a, b)  # inverse CDF of Kumaraswamy
        return x

    @staticmethod
    def moment_n(a, b, n):
        return b * beta_f(1. + n / a, b)

    @staticmethod
    def kumaraswamy_nll(params, data):
        """
        Negative log-likelihood for Kumaraswamy(a,b)
        data: [N] array with values in (0,1)
        params: [log(a), log(b)]
        """
        loga, logb = params
        a = np.exp(loga)
        b = np.exp(logb)

        # Numerical safety
        eps = 1e-12
        x = np.clip(data, eps, 1 - eps)

        log_pdf = loga + logb + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)
        reg_a, reg_b = 0.0, 1.

        return -np.mean(log_pdf) + reg_b * logb + reg_a * loga

    @staticmethod
    def MLE_Kumaraswamy_1d(data, init=(0.0, 0.0)):
        """
        Fit univariate Kumaraswamy distribution by MLE.
        Returns (a, b)
        """
        res = minimize(
            KumaraswamyMultivariate.kumaraswamy_nll,
            x0=np.array(init),
            args=(data,),
            method="L-BFGS-B",
            bounds=[(1e-4, np.log(1e5)), (1e-4, np.log(1e5))],
        )
        if not res.success:
            raise RuntimeError(f"MLE failed: {res.message}")
        a, b = np.exp(res.x)
        return a, b

    def MLE_Kumaraswamy_multi(self, data):
        """
        Estimate parameters (a_d, b_d) for D-dimensional Kumaraswamy distribution.
        Each dimension is treated independently.

        Args:
            data: [N, D] array, values in (0,1)
        Returns:
            a: [D]
            b: [D]
        """
        N, D = data.shape
        a_hat = np.zeros(D)
        b_hat = np.zeros(D)
        init_a, init_b = np.log(self._a), np.log(self._b)
        for d in range(D):
            try:
                a_hat[d], b_hat[d] = self.MLE_Kumaraswamy_1d(data[:, d], (init_a[d], init_b[d]))
            except RuntimeError:
                a_hat[d], b_hat[d] = self._a[d], self._b[d]
        return a_hat, b_hat

    def MME(self, data):
        """
        Estimate (a, b) for each dimension of data ~ Kumaraswamy(a, b)
        """
        data = np.asarray(data)
        N, D = data.shape
        
        x_bar = np.mean(data, axis=0)
        x2_bar = np.mean(data**2, axis=0)

        a_hat = np.zeros(D)
        b_hat = np.zeros(D)

        # Solve each dimension independently
        for d in range(D):
            mean_d = x_bar[d]
            var_d = x2_bar[d] - mean_d**2

            def equations(log_params):
                a, b = np.exp(log_params)
                m1 = self.moment_n(a, b, 1) - mean_d
                m2 = (self.moment_n(a, b, 2) - self.moment_n(a, b, 1)**2) - var_d
                return [m1, m2]

            sol = root(equations, [np.log(self._a[d]), np.log(self._b[d])], method="lm")
            if not sol.success:
                print(f"Warning: solver failed for dim {d}: {sol.message}")
                a_hat[d], b_hat[d] = np.nan, np.nan
            else:
                a_hat[d], b_hat[d] = np.exp(sol.x)

        return a_hat, b_hat

    def estimate_params(self, samples):
        """
        Estimate marginal parameters a, b, and correlation matrix Sigma.
        samples: [N, D]
        """
        a, b = self.MLE_Kumaraswamy_multi(samples)

        u = self.cdf(samples, a, b)
        z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        Sigma = np.corrcoef(z, rowvar=False)

        return a, b, Sigma
    

AVAILABLE_SAMPLERS = {
    "normal": MultivariateNormal,
    "beta": BetaMultivariateCopulas,
    "kumaraswamy": KumaraswamyMultivariate,
}