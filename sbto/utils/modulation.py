import numpy as np
from scipy.stats import beta


def beta_mod(it: int, Nit: int, T: int, b_start: float = 1.5, **kwargs):
    b_end = 0.
    a = 1.

    b = np.logspace(b_start, b_end, Nit, endpoint=True)[it]
    t = np.arange(T) / T
    mod = beta.pdf(t, a, b)

    return mod

def step_mod(it: int, Nit: int, T: int, Nknots: int, **kwargs):
    n_step = np.int32(np.linspace(0, 1 , Nit+1) * (Nknots-1))[it] + 1
    mod = np.zeros(T)
    i = np.int32(np.linspace(0, T, Nknots, endpoint=True))[n_step]
    mod[:i] = 1.
    mod /= np.sum(mod)
    mod *= T
    return mod

def mpc_mod(it: int, Nit: int, T: int, Nknots: int, **kwargs):
    n_step = np.int32(np.linspace(0, 1 , Nit+1) * (Nknots-1))[it] + 1
    mod = np.zeros(T)
    i = np.int32(np.linspace(0, T-1, Nknots, endpoint=True))[n_step-1:n_step+1]
    i = np.clip(i, 0, T-1)
    mod[slice(*i)] = 1.
    mod /= np.sum(mod)
    mod *= T
    return mod

def step_decayed_mod(it: int, Nit: int, T: int, Nknots: int, decay_rate: float = 0.5, **kwargs):
    mod = np.zeros(T)
    n_step = np.int32(np.linspace(0, 1 , Nit+1) * (Nknots-1))[it] + 1

    for i in range(n_step):
        s = np.int32(np.linspace(0, T, Nknots, endpoint=True))[i:i+2]
        decay = np.exp(-decay_rate*(n_step-i)/n_step)
        mod[slice(*s)] = decay

    mod /= np.sum(mod)
    mod *= T
    return mod

def step_mod_transition(it: int, Nit: int, T: int, Nknots: int, **kwargs):
    n_step = np.int32(np.linspace(0, 1 , Nit+1) * (Nknots-1))[it] + 1
    mod = np.zeros(T)
    i = np.int32(np.linspace(0, T, Nknots, endpoint=True))[n_step]
    mod[:i] = 1.

    if n_step < Nknots - 1:
        i_transition = np.int32(np.linspace(0, T, Nknots, endpoint=True))[n_step+1]
        v = (it % (Nit // Nknots)) / Nknots
        mod[i:i_transition] = (it % (Nknots)) / Nknots

    mod /= np.sum(mod)
    mod *= T
    return mod