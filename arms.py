""" Packages import """
import numpy as np
from scipy.stats import truncnorm as trunc_norm
from scipy.stats import norm, gennorm
from utils import convert_tg_mean


class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        :param mean: float, expectation of the arm
        :param variance: float, variance of the arm
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance
        self.local_random = np.random.RandomState(random_state)

    def sample(self, size=1):
        pass


class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, mean parameter
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state
                                           )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return (self.local_random.rand(size) < self.p) * 1.


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        :param a: int, alpha coefficient in beta distribution
        :param b: int, beta coefficient in beta distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a / (a + b),
                                      variance=(a * b) / ((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state
                                      )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.beta(self.a, self.b, size)


class ArmGaussian(AbstractArm):
    def __init__(self, mu, eta, random_state=0):
        """
        :param mu: float, mean parameter in Gaussian distribution
        :param eta: float, std parameter in Gaussian distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta
        super(ArmGaussian, self).__init__(mean=mu,
                                          variance=eta ** 2,
                                          random_state=random_state
                                          )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.normal(self.mu, self.eta, size)


class ArmLogGaussian(AbstractArm):
    def __init__(self, mu, eta, random_state=0):
        """
        :param mu: float, mean parameter in log-Gaussian distribution
        :param eta: float, std parameter in log-Gaussian distribution
        :param alpha: float or None, risk aversion level
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta
        super(ArmLogGaussian, self).__init__(mean=np.exp(mu + 0.5 * eta ** 2),
                                             variance=(np.exp(eta ** 2) - 1) * np.exp(2 * mu + eta ** 2),
                                             random_state=random_state,
                                             )

    def sample(self, size=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.lognormal(self.mu, self.eta, size)


class ArmMultinomial(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        :param X: np.array, support of the distribution
        :param P: np.array, associated probabilities
        :param random_state: int, seed to make experiments reproducible
        """
        assert np.min(P) >= 0.0, 'p should be nonnegative.'
        assert np.isclose(np.sum(P), 1.0), 'p should should sum to 1.'

        self.X = np.array(X)
        self.P = np.array(P)
        mean = np.dot(self.X, self.P)
        super(ArmMultinomial, self).__init__(mean=mean,
                                             variance=np.dot(self.X ** 2, self.P) - mean ** 2,
                                             random_state=random_state
                                             )

    def sample(self, size=1):
        """
        Sampling strategy for an arm with a finite support and the associated probability distribution
        :return: float, a sample from the arm
        """
        i = self.local_random.choice(len(self.P), size=size, p=self.P)
        reward = self.X[i]
        return reward


class ArmEmpirical(AbstractArm):
    def __init__(self, X, random_state=0):
        """
        Same a ArmMultinomial but with uniform probability.
        Allows for faster sampling using randint rather than choice
        (~x4 speed up).
        :param X: np.array, support of the distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.X = np.array(X)
        self.n = len(self.X)
        mean = np.mean(self.X)
        super(ArmEmpirical, self).__init__(mean=mean,
                                           variance=np.var(self.X),
                                           random_state=random_state
                                           )

    def sample(self, size=1):
        """
        Sampling strategy for an arm with a finite support and the associated probability distribution
        :return: float, a sample from the arm
        """
        i = self.local_random.randint(len(self.X), size=size)
        reward = self.X[i]
        return reward


class ArmExponential(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, parameter in exponential distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmExponential, self).__init__(mean=p,
                                             variance=p ** 2,
                                             random_state=random_state
                                             )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.exponential(self.p, size)


class ArmDirac():
    def __init__(self, c, random_state):
        """
        :param c: float, location of the mass
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = c
        self.variance = 0
        self.local_random = np.random.RandomState(random_state)

    def sample(self, size=1):
        return self.mean * np.ones(size)


class ArmTG(AbstractArm):
    def __init__(self, mu, scale, random_state=0):
        """
        :param mu: float, mean of the untruncated Gaussian
        :param scale: float, std of the untruncated Gaussian
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.scale = scale
        self.dist = trunc_norm(-mu / scale, b=(1 - mu) / scale, loc=mu, scale=scale)
        self.dist.random_state = random_state
        super(ArmTG, self).__init__(mean=convert_tg_mean(mu, scale),
                                    variance=scale ** 2,
                                    random_state=random_state
                                    )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        x = self.local_random.normal(self.mu, self.scale, size)
        return x * (x > 0) * (x < 1) + (x > 1)


class ArmPoisson(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, Poisson parameter
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmPoisson, self).__init__(mean=p,
                                         variance=None,
                                         random_state=random_state
                                         )

    def sample(self, size):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.poisson(self.p, size=size)


class ArmNegativeExponential(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, parameter in exponential distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmNegativeExponential, self).__init__(mean=-p,
                                                     variance=p ** 2,
                                                     random_state=random_state
                                                     )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return -self.local_random.exponential(self.p, size)


class ArmPareto(AbstractArm):
    def __init__(self, alpha, C, random_state=0):
        """
        :param alpha: float, exponent in Pareto distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.alpha = alpha
        self.scale = C ** (1 / alpha)
        super(ArmPareto, self).__init__(mean=alpha / (alpha - 1) * C ** (1 / alpha),
                                        variance=None,
                                        random_state=random_state
                                        )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return (self.local_random.pareto(self.alpha, size) + 1) * self.scale


class ArmGaussianMixture(AbstractArm):
    def __init__(self, p, means, sigmas, random_state=0):
        """
        :param p: array, probability of each Gaussian in the mixture
        :param means: array, means for each Gaussian in the mixture
        :param sigmas: array, stds for each Gaussian in the mixture
        :param random_state: int, seed to make experiments reproducible
        """
        assert np.min(p) >= 0.0, 'p should be nonnegative.'
        assert np.isclose(np.sum(p), 1.0), 'p should should sum to 1.'
        self.p = np.array(p)
        self.means = np.array(means)
        self.sigmas = np.array(sigmas)
        super(ArmGaussianMixture, self).__init__(mean=np.inner(self.p, self.means),
                                                 variance=np.inner(self.p, self.sigmas**2),
                                                 random_state=random_state
                                                 )

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        i = self.local_random.choice(np.arange(len(self.p)), p=self.p, size=size)
        return self.local_random.normal(loc=self.means[i], scale=self.sigmas[i], size=size)


class ArmUniform(AbstractArm):
    def __init__(self, low, high, random_state=0):
        """
        :param low: float, lower bound of support
        :param high: float, upper bound of support
        :param random_state: int, seed to make experiments reproducible
        """
        self.low = low
        self.high = high
        super(ArmUniform, self).__init__(mean=0.5 * (low + high),
                                         variance=1 / 12 * (high - low) ** 2,
                                         random_state=random_state
                                         )

    def sample(self, size=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.uniform(low=self.low, high=self.high, size=size)


class ArmGenNorm(AbstractArm):
    def __init__(self, beta, random_state=0):
        """
        :param mu: float, loc parameter for distribution
        :param eta: float, scale parameter for distribution
        :param asymmetry: float, asymmetry parameter for the impostor Laplace distribution
        :param misspecified: bool, whether or not to contaminate the tail
        :param random_state: int, seed to make experiments reproducible
        """
        self.beta = beta
        super(ArmGenNorm, self).__init__(mean=None,
                                         variance=None, #gennorm.stats(beta, 'v'),
                                         random_state=random_state)

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return gennorm(self.beta).rvs(size)


class ArmMixture(AbstractArm):
    def __init__(self, arms_type, probabilities, param_list, random_state=0):
        """
        :param params: list [arms type, mixture probabilities, parameters list]
        :param random_state: int, seed to make experiments reproducible
        """
        self.arms_type = arms_type
        self.nb_subarms = len(arms_type)
        self.prob = probabilities
        self.p = param_list
        self.arms = []
        self.build()
        super(ArmMixture, self).__init__(mean=None,
                                         variance=None,
                                         random_state=random_state
                                         )

    def build(self):
        for i, m in enumerate(self.arms_type):
            args = [self.p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(self.p[i]) == list else args
            alg = mapping[m]
            self.arms.append(alg(*args))

    def sample(self, size=1):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        k = np.random.choice(np.arange(self.nb_subarms), p=self.prob, replace=True, size=size)
        return np.array([self.arms[i].sample()[0] for i in k])


mapping = {
    'B': ArmBernoulli,
    'Beta': ArmBeta,
    'Dirac': ArmDirac,
    'Emp': ArmEmpirical,
    'Exp': ArmExponential,
    'G': ArmGaussian,
    'GMx': ArmGaussianMixture,
    'LG': ArmLogGaussian,
    'M': ArmMultinomial,
    'NegExp': ArmNegativeExponential,
    'P': ArmPoisson,
    'Par': ArmPareto,
    'TG': ArmTG,
    'U': ArmUniform,
    'GenNorm': ArmGenNorm
}