import numpy as np
from scipy.stats import beta


def beta_decay(it: int, Nit: int, T: int, b_start: float = 1.5):
    b_end = 0.
    a = 1.

    b = np.logspace(b_start, b_end, Nit, endpoint=True)[it]
    t = np.arange(T) / T
    decay = beta.pdf(t, a, b)

    return decay

def step_decay(it: int, Nit: int, T: int, Nknots: int):
    n_step = np.int32(np.linspace(0, 1 , Nit+1) * (Nknots-1))[it] + 1
    decay = np.zeros(T)
    i = np.int32(np.linspace(0, T, Nknots))[n_step]
    decay[:i] = 1.
    decay /= np.sum(decay)
    decay *= T
    return decay