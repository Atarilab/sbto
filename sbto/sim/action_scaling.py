import numpy as np
from abc import ABC, abstractmethod

class ScalerAbstract(ABC):
    """Base class for scaling functions between normalized and physical spaces."""
    
    def __init__(self,
                 q_min,
                 q_max,
                 q_nom = None,
                 clip = False,
                 **kwargs):
        self.q_min = q_min
        self.q_max = q_max
        self.q_nom = q_nom if q_nom is not None else 0.5 * (q_min + q_max)
        self.clip = clip
        self.kwargs = kwargs

    @abstractmethod
    def _scale(self, act: np.ndarray) -> np.ndarray:
        """Scale from normalized action space → physical space."""
        return act

    def scale(self, act: np.ndarray) -> np.ndarray:
        """Scale from normalized action space → physical space."""
        act_scaled = self._scale(act)
        if self.clip:
            act_scaled = np.clip(act_scaled, self.q_min, self.q_max)
        return act_scaled

    @abstractmethod
    def inverse(self, q: np.ndarray) -> np.ndarray:
        """Inverse scale from physical space → normalized space."""
        return q

# === Derived Scaling Classes ===
class NominalScaling(ScalerAbstract):
    """Piecewise linear asymmetric scaling."""

    def _scale(self, act):
        return act + self.q_nom

    def inverse(self, q):
        return q - self.q_nom

class AsymmetricScaling(ScalerAbstract):
    """Piecewise linear asymmetric scaling."""

    def _scale(self, act):
        return self.q_nom + np.where(
            act < 0,
            act * (self.q_nom - self.q_min),
            act * (self.q_max - self.q_nom)
        )

    def inverse(self, q):
        return np.where(
            q < self.q_nom,
            (q - self.q_nom) / (self.q_nom - self.q_min),
            (q - self.q_nom) / (self.q_max - self.q_nom)
        )

class SmoothAsymmetricScaling(ScalerAbstract):
    """Smooth asymmetric scaling using logistic blending."""

    def __init__(self, q_min, q_max, q_nom=None, clip=False, act_scale=10., **kwargs):
        super().__init__(q_min, q_max, q_nom, **kwargs)
        self.act_scale = act_scale

    def _scale(self, act):
        s = 1. / (1. + np.exp(-self.act_scale * act))
        scale = (self.q_nom - self.q_min) * (1. - s) + (self.q_max - self.q_nom) * s
        return self.q_nom + act * scale

    def inverse(self, q):
        # Approximate inverse via Newton iteration
        q = np.clip(q, self.q_min, self.q_max)
        act = np.zeros_like(q)
        for _ in range(5):  # few fixed-point iterations
            s = 1. / (1. + np.exp(-self.act_scale * act))
            scale = (self.q_nom - self.q_min) * (1. - s) + (self.q_max - self.q_nom) * s
            f = self.q_nom + act * scale - q
            df = scale + act * self.act_scale * s * (1. - s) * ((self.q_max - self.q_nom) - (self.q_nom - self.q_min))
            act -= f / (df + 1e-8)
        return np.clip(act, -1., 1.)

class TanhScaling(ScalerAbstract):
    """Smooth symmetric scaling using tanh."""

    def __init__(self, q_min, q_max, q_nom=None, clip=False, act_scale=10., **kwargs):
        super().__init__(q_min, q_max, q_nom, **kwargs)
        self.act_scale = act_scale

    def _scale(self, act):
        return self.q_nom + 0.5 * (self.q_max - self.q_min) * np.tanh(self.act_scale * act)

    def inverse(self, q):
        q = np.clip(q, self.q_min, self.q_max)
        y = 2 * (q - self.q_nom) / (self.q_max - self.q_min)
        y = np.clip(y, -0.999999, 0.999999)
        return np.arctanh(y) / self.act_scale

class LinearScaling01(ScalerAbstract):
    """Linear symmetric scaling from [0,1] → [q_min,q_max]."""

    def _scale(self, act):
        return self.q_min + act * (self.q_max - self.q_min)

    def inverse(self, q):
        q = np.clip(q, self.q_min, self.q_max)
        return (q - self.q_min) / (self.q_max - self.q_min)


class LinearScaling11(ScalerAbstract):
    """Linear symmetric scaling from [-1,1] → [q_min,q_max]."""

    def _scale(self, act):
        return 0.5 * (self.q_max + self.q_min) + 0.5 * act * (self.q_max - self.q_min)

    def inverse(self, q):
        q = np.clip(q, self.q_min, self.q_max)
        return 2 * (q - 0.5 * (self.q_max + self.q_min)) / (self.q_max - self.q_min)

class Scaling():
    def __init__(self,
                 name: str = "",
                 clip: bool = False,
                 **kwargs,
                 ):
        self.clip = clip
        self.kwargs = kwargs

        AVAILABLE_SCALING = {
            "": ScalerAbstract,
            "nominal": NominalScaling,
            "asymmetric": AsymmetricScaling,
            "smooth_asymmetric": SmoothAsymmetricScaling,
            "tanh": TanhScaling,
            "linear": LinearScaling01,
            "linear11": LinearScaling11,
        }

        if name == "" or not name in AVAILABLE_SCALING.keys():
            self.scaler = ScalerAbstract(0, 0, 0, False)
        else:
            self.scaler_class = AVAILABLE_SCALING[name]
            self.scaler = None

    def set_range(self, q_min, q_max, q_nom):
        self.scaler = self.scaler_class(q_min, q_max, q_nom, self.clip, **self.kwargs)

    def _check_scaler_init(self):
        if self.scaler is None:
            raise ValueError("Scaler function is not properly initialized. Call set_range first.")

    def __call__(self, act):
        return self.scaler.scale(act)
    
    def inverse(self, q):
        return self.scaler.inverse(q)