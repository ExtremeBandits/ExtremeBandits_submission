""" Packages import """
import numpy as np
from numba import jit
import scipy.stats as sc
from scipy.special import gamma
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@jit(nopython=True)
def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


@jit(nopython=True)
def rd_choice(vec, size):
    return np.random.choice(vec, size=size, replace=False)


@jit(nopython=True)
def get_leader_qomax(n, qomax):
    m = np.amax(n)
    n_argmax = np.nonzero(n == m)[0]
    if n_argmax.shape[0] == 1:
        return n_argmax[0]
    else:
        maximomax = qomax[n_argmax].max()
        s_argmax = np.nonzero(qomax[n_argmax] == maximomax)[0]
    return n_argmax[np.random.choice(s_argmax)]


def convert_tg_mean(mu, scale, step=1e-7):
    X = np.arange(0, 1, step)
    return (X * sc.norm.pdf(X, loc=mu, scale=scale)).mean() + 1 - sc.norm.cdf(1, loc=mu, scale=scale)


def kl(p, q):
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))


def Chernoff_Interval(mu, n, alpha):
    """
    U function for threshold ascent
    """
    if n == 0:
        return np.inf
    return mu + (alpha + np.sqrt(2*n*mu*alpha+alpha**2))/n


def second_order_Pareto_UCB(tr, b, D, E, delta, r):
    """
    tr: TrackerMax object
    b: speed of convergence to a Pareto distrib (see Carpentier et Valko)
    D: constant scaling the bonus for the constant
    E: constant scaling of the bonus for the rate
    delta: error allowed
    r: fraction of sorted samples used for parameter estimation
    """
    B1 = D * np.sqrt(np.log(1 / delta)) * tr.Na ** (-b / (2 * b + 1))
    B2 = E * np.sqrt(np.log(tr.Na / delta)) * np.log(tr.Na) * tr.Na ** (-b / (2 * b + 1))
    hk = np.zeros(tr.nb_arms)
    Ck = np.zeros(tr.nb_arms)
    Bk = np.zeros(tr.nb_arms)
    for k in range(tr.nb_arms):
        s = int(r * tr.Na[k])
        rwd = np.array(np.maximum(tr.sorted_rewards_arm[k], 1))
        hk[k] = np.log(rwd[-s:] / rwd[-s]).mean()
        Ck[k] = tr.Na[k] ** (1 / (2 * b + 1)) * (rwd >= tr.Na[k] ** (hk[k] / (2 * b + 1))).mean()
        if hk[k] + B1[k] >= 1:
            Bk[k] = np.inf
        else:
            Bk[k] = ((Ck[k] + B2[k]) * tr.T) ** (hk[k] + B1[k]) * gamma(1 - hk[k] - B1[k])
    return int(rd_argmax(Bk))
