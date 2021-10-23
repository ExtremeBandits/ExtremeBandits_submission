""" Packages import """
import numpy as np
import arms
from tqdm import tqdm
from utils import rd_argmax
from utils import Chernoff_Interval, get_leader_qomax, second_order_Pareto_UCB
from tracker import TrackerMax
from bisect import bisect

mapping = {
    'B': arms.ArmBernoulli,
    'Beta': arms.ArmBeta,
    'Dirac': arms.ArmDirac,
    'Emp': arms.ArmEmpirical,
    'Exp': arms.ArmExponential,
    'G': arms.ArmGaussian,
    'LG': arms.ArmLogGaussian,
    'GenNorm': arms.ArmGenNorm,
    'M': arms.ArmMultinomial,
    'NegExp': arms.ArmNegativeExponential,
    'P': arms.ArmPoisson,
    'Par': arms.ArmPareto,
    'TG': arms.ArmTG,
    'U': arms.ArmUniform,
    'Mixture': arms.ArmMixture
    }


class GenericMAB:
    """
    Generic class to simulate an Extreme Bandit problem
    """
    def __init__(self, methods, p):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        """
        self.arm_types = methods
        self.params = p
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.mc_regret = None

    @staticmethod
    def generate_arms(methods, p):
        """
        Method for generating the frozen distribution
        corresponding to each arm, according to the given parameters
        :param methods: string, probability distribution of each arm (general family)
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(methods):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            alg = mapping[m]
            arms_list.append(alg(*args))
        return arms_list

    def MC(self, method, N, T, param_dic):
        """
        Implementation of Monte Carlo method: sample N trajectories for horizon T and
        store the results
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent trajectories
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods,
        ex: {'param1': value1, 'param2': value2}
        """
        mc_count = np.zeros(self.nb_arms)
        all_counts = np.zeros((N, self.nb_arms))
        all_maxima = np.zeros(N)
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            tr = alg(T, **param_dic)
            mc_count += tr.Na/N
            all_counts[i] = tr.Na
            all_maxima[i] = tr.current_max
        return mc_count, all_counts, all_maxima

    def ETC(self, T, size, func, rwd_arm=True, sorted_rwd=False):
        """
        Generic Explore-Then-Commit strategy
        :param T:  int, time horizon
        :param size: sample size required for each arm in the exploration phase
        :param func: decision rule for the end of the exploration phase
        :return: means, arm sequence
        """
        tr = TrackerMax(self.nb_arms, T, store_rewards_arm=rwd_arm,
                        store_sorted_rewards_arm=sorted_rwd)
        if size * self.nb_arms > T:
            return self.Uniform_Exploration(T)
        t = 0
        while t < min(self.nb_arms * size, T):
            arm = t % self.nb_arms
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        arm_final = func(tr)
        while t < T:
            tr.update(t, arm_final, self.MAB[arm_final].sample()[0])
            t += 1
        return tr

    def Uniform_Exploration(self, T):
        """
        Implementation of a policy sampling each arm in the exact same proportions
        :param T:  int, time horizon
        :return: means, arm sequence
        """
        tr = TrackerMax(self.nb_arms, T, store_rewards_arm=False)
        for t in range(T):
            arm = t % self.nb_arms
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def Threshold_Ascent(self, T, s, delta):
        """
        Threshold Ascent strategy (Streeter et al., 2006)
        :param T:  int, time horizon
        :param s: number of data kept by the algorithm
        :param delta: heuristic for a "probability of mistake", number between 0 and 1
        :return: means, arm sequence
        """
        tr = TrackerMax(self.nb_arms, T, store_all_sorted_rewards=True, store_rewards_arm=True)
        alpha = np.log(2*self.nb_arms*T/delta)
        threshold = -np.inf
        for t in range(T):
            if t > s:
                former_thresh = threshold
                threshold = tr.all_sorted_rewards[-int(s)]
                if threshold != former_thresh:
                    S = [np.array(tr.rewards_arm[k])[np.array(tr.rewards_arm[k]) >= threshold].shape[0]
                         for k in range(self.nb_arms)]
            else:
                S = tr.Na
            Idx = np.array([Chernoff_Interval(S[k]/tr.Na[k], tr.Na[k], alpha)
                            for k in range(self.nb_arms)])
            arm = rd_argmax(Idx)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def ExtremeHunter(self, T, b=100, D=1e-4, E=1e-4, N=None, r=None, steps=1, delta=0.1):
        """
        Implementation of ExtremeHunter (Carpentier & Valko, 2013)
        :param T:  int, time horizon
        :param b: flot, parameter of the second-order Pareto condition
        :param D: parameter of the UCB on the "rate" (Hill estimator of lambda)
        :param E: parameter of the UCB on the "scale" of the tail
        :param N: number of times each arm is pulled during the initial exploration phase,
        if None the theoretical value (Carpentier & Valko) is used (dependent on T)
        :param r: number of data kept to compute the estimators,
        if 'theoretic' the theoretical value (Carpentier & Valko) is used (depend on T, b)
        :param steps:, to avoid computational burden for large T, compute a second-order
        pareto UCB only after pullings "steps" data.
        :param delta: probability of mistake, if None use the theoretical value.
        Not the default parameter has it is huge for reasonable horizons.
        :return: means, arm sequence
        """
        tr = TrackerMax(self.nb_arms, T, store_sorted_rewards_arm=True)
        if r is None:
            r = T**(-1/(2*b+1))  # from Carpentier & Valko
        if N is None:
            N = np.log(T) ** ((2*b+1)/b)  # from Carpentier & Valko
        if delta == 'theoretic':
            delta = np.exp(-np.log(T)**2)/(2*T*self.nb_arms)  # theoretic value from paper, 0.1 used in their code
        t = 0
        while t < T:
            if t < self.nb_arms * N:  # First exploration phase
                arm = t % self.nb_arms
                tr.update(t, arm, self.MAB[arm].sample()[0])
                t += 1
            else:
                arm = second_order_Pareto_UCB(tr, b, D, E, delta, r)
                nb_pulls = min(T-t, steps)
                for _ in range(nb_pulls):
                    tr.update(t, arm, self.MAB[arm].sample()[0])
                    t += 1
        return tr

    def ExtremeETC(self, T, b=100, D=1e-4, E=1e-4, delta=0.1, N=None, r=None):
        """
        Same parameters as ExtremeHunter.
        """
        if r is None:
            r = T**(-1/(2*b+1))  # from code Carpentier & Valko
        if N is None:
            N = np.log(T) ** ((2*b+1)/b)  # from code Carpentier & Valko
        if delta == 'theoretic':
            delta = np.exp(-np.log(T) ** 2) / (2 * T * self.nb_arms)
        def f(tr):
            return second_order_Pareto_UCB(tr, b, D, E, delta, r)
        return self.ETC(T, N, f, rwd_arm=False, sorted_rwd=True)

    def simple_ETC(self, T, size):
        """
        Just an ETC strategy being greedy, we used it as a baseline in our
        first experiments to see if QoMax was indeed performing better (this
        is clearly the case, which motivated investigating theoretical properties of QoMax)
        :param T: Time Horizon
        :param size:
        """
        def f(tr):
            return int(rd_argmax(tr.max_arms))
        return self.ETC(T, size, f)

    def MaxMedian(self, T, explo_func):
        """
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
        """
        tr = TrackerMax(self.nb_arms, T, store_sorted_rewards_arm=True)
        t = 0
        while t < self.nb_arms:
            tr.update(t, t, self.MAB[t].sample()[0])
            t += 1
        while t < T:
            if np.random.binomial(1, explo_func(t)) == 1:
                k = np.random.randint(self.nb_arms)
            else:
                m = tr.Na.min()
                orders = np.ceil(tr.Na/m).astype(np.int32)
                idx = [tr.sorted_rewards_arm[i][-orders[i]] for i in range(self.nb_arms)]
                k = rd_argmax(np.array(idx))
            reward = self.MAB[k].sample()[0]
            tr.update(t, k, reward)
            t += 1
        return tr

    def QoMax_ETC(self, T, size, q):
        """
        Implementation of QoMax-ETC
        :param T: Time Horizon
        :param size: tuple, (number of batches, number of samples per batch)
        :param q: quantile used
        """
        def f(tr):
            qomax = np.zeros(self.nb_arms)
            for k in range(self.nb_arms):
                samples = np.array(tr.rewards_arm[k]).reshape(size)
                M = np.max(samples, axis=1)
                qomax[k] = np.quantile(M, q)
            return int(rd_argmax(qomax))
        return self.ETC(T, size[0]*size[1], f)

    def compute_qomax_dic(self, mx, q):
        """
        Computation of the QoMax using the storage trick
        """
        return np.quantile([list(mx[i].values())[0] for i in range(mx.__len__())], q)

    def compute_qomax_list(self, l, q):
        return np.quantile(l, q)

    def qomax_duel(self, tr, l, k, chosen_arms_prev, fe, q=None):
        if k == l:
            return k
        if tr.n[k] <= fe(tr.r):
            return k
        # Compute leader's QoMax (on Last Block subsample)
        l_max = np.zeros(tr.nb_batch[k])
        for i in range(tr.nb_batch[k]):
            last_idx = tr.n[l] - tr.n[k]
            idx_dic = [*tr.maxima[l][i]]
            l_max[i] = tr.maxima[l][i][idx_dic[bisect(idx_dic, last_idx)]]
        sub_qomax = self.compute_qomax_list(l_max, q)
        # Comparison with challenger's MoMax
        if k in chosen_arms_prev or tr.qomax[k] == np.inf:
            tr.qomax[k] = self.compute_qomax_dic(tr.maxima[k], q)  # Update if MoMax changes
        if sub_qomax <= tr.qomax[k]:
            return k

    def QoMax_SDA(self, T, fe, batch_size, q=0.5):
        """
        Implementation of the QoMaX-SDA strategy
        :param T: time Horizon
        :param fe: sampling obligation (number of queries required at a given round)
        :param batch_size: function defining the number of batches required for a give
        number of queries
        :param q: quantile
        """
        tr = TrackerMax(self.nb_arms, T, store_maxima=True)
        chosen_arms, l_prev = [-2], -1
        tr.r = 1
        while tr.t < T:
            if chosen_arms == [l_prev]:
                l = l_prev
            else:
                l = get_leader_qomax(tr.n, tr.qomax)  # Compute_leader
                l_prev = l
            chosen_arms_prev = [x for x in chosen_arms]
            chosen_arms = []
            for k in range(self.nb_arms):  # Duel step
                if self.qomax_duel(tr, l, k, chosen_arms_prev, fe, q) == k and k != l:
                    chosen_arms.append(k)
            if tr.n[l] <= fe(tr.r):
                chosen_arms.append(l)
            if len(chosen_arms) == 0:
                chosen_arms = [l]
            tr.collect_rewards_batch(self.MAB, chosen_arms, l, batch_size)
            tr.r += 1
        tr.compute_max_from_dic()
        return tr
