import numpy as np

# all scaling on array "act" of shape [N, Nknots, Nu]

def asymmetric_scaling(act, q_min, q_max, q_nom, **kwargs):
    """
    Piecewise linear asymmetric scaling.
    Scales 'act' ∈ [-1,1] around q_nom, using different slopes
    toward q_min and q_max.
    """
    act = np.clip(act, -1., 1.)
    return q_nom + np.where(
        act < 0,
        act * (q_nom - q_min),
        act * (q_max - q_nom)
    )

def smooth_asymmetric_scaling(act, q_min, q_max, q_nom, act_scale=10., **kwargs):
    """
    Smooth asymmetric scaling using a logistic blending of slopes.
    Acts like a differentiable version of asymmetric_scaling.
    """
    s = 1. / (1. + np.exp(-act_scale * act))  # smooth blend between sides
    scale = (q_nom - q_min) * (1. - s) + (q_max - q_nom) * s
    return q_nom + act * scale


def tanh_scaling(act, q_min, q_max, q_nom, act_scale=10., **kwargs):
    """
    Smooth symmetric scaling using tanh.
    Ignores q_nom (centered midrange). For asymmetric ranges,
    q_nom can be used as offset if desired.
    """
    return q_nom + 0.5 * (q_max - q_min) * np.tanh(act_scale * act)


def linear_scaling_01(act, q_min, q_max, q_nom, **kwargs):
    """
    Simple linear symmetric scaling from [0,1] → [q_min,q_max].
    """
    act = np.clip(act, 0., 1.)
    return q_min + act * (q_max - q_min)

def linear_scaling_11(act, q_min, q_max, q_nom, **kwargs):
    """
    Simple linear symmetric scaling from [-1,1] → [q_min,q_max].
    """
    act = np.clip(act, -1., 1.)
    return 0.5 * (q_max + q_min) + 0.5 * act * (q_max - q_min)

AVAILABLE_SCALING = {
    "asymmetric" : asymmetric_scaling,
    "smooth_asymmetric" : smooth_asymmetric_scaling,
    "tanh" : tanh_scaling,
    "linear" : linear_scaling_01,
    "linear11" : linear_scaling_11,
}