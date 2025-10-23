import numpy as np
import matplotlib.pyplot as plt

def cov_cauchy(Nu: int,
                Nknots: int,
                l: float = 1.0,
                exp: float = 2.,
                sigma: float = 1.0):
    N = Nu * Nknots
    cov = np.zeros((N, N))
    t = np.arange(Nknots)
    base_cov = sigma**2 * (1 + np.abs(np.subtract.outer(t, t)) / l) ** (-exp)
    for nu in range(Nu):
        cov[nu::Nu, nu::Nu] = base_cov
    return cov

def cov_rbf(Nu, Nknots, l=1.0, sigma=1.0):
    N = Nu * Nknots
    t = np.arange(Nknots)
    d = np.subtract.outer(t, t)
    base_cov = sigma**2 * np.exp(-0.5 * (d / l)**2)
    cov = np.zeros((N, N))
    for nu in range(Nu):
        cov[nu::Nu, nu::Nu] = base_cov
    return cov

def cov_exponential(Nu, Nknots, l=1.0, sigma=1.0):
    N = Nu * Nknots
    t = np.arange(Nknots)
    d = np.abs(np.subtract.outer(t, t))
    base_cov = sigma**2 * np.exp(-d / l)
    cov = np.zeros((N, N))
    for nu in range(Nu):
        cov[nu::Nu, nu::Nu] = base_cov
    return cov

def cov_matern32(Nu, Nknots, l=1.0, sigma=1.0):
    N = Nu * Nknots
    t = np.arange(Nknots)
    d = np.abs(np.subtract.outer(t, t))
    base_cov = sigma**2 * (1 + np.sqrt(3) * d / l) * np.exp(-np.sqrt(3) * d / l)
    cov = np.zeros((N, N))
    for nu in range(Nu):
        cov[nu::Nu, nu::Nu] = base_cov
    return cov


if __name__ == "__main__":
    
    Nu = 1
    Nknots = 20

    for c in [cov_cauchy, cov_rbf, cov_exponential, cov_matern32]:
        cov = c(Nu, Nknots, l=Nknots/3., sigma=1.)
        plt.imshow(cov)
        plt.show()

    # a, b = 0., 1.
    # Id = np.diag(np.linspace(a, b, Nknots).repeat(Nu))
    # plt.imshow(Id)
    # plt.show()

    mean = np.zeros(int( Nu * Nknots ))
    sample = np.random.multivariate_normal(mean, cov)

    plt.plot(sample.reshape(Nknots, Nu))
    plt.show()