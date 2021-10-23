import numpy as np
from numba import float64, int64
from bisect import insort, bisect


spec = [
    ('means', float64[:]),
    ('nb_arms', int64),
    ('T', int64),
    ('Sa', float64[:]),
    ('Na', int64[:]),
    ('Sa_bin', int64[:]),
    ('Na_bin', int64[:]),
    ('reward', float64[:]),
    ('arm_sequence', int64[:]),
    ('t', int64)
]


class Tracker2:
    def __init__(self,
                 means,
                 T,
                 store_rewards_arm=False,
                 store_max_rewards_arm=False,
                 store_sorted_rewards_arm=False,
                 store_all_sorted_rewards=False,
                 q_left=None,
                 q_right=None,
                 binarization_trick=False,
                 ):
        self.means = means
        self.nb_arms = means.shape[0]
        self.T = T
        self.store_rewards_arm = store_rewards_arm
        self.store_max_rewards_arm = store_max_rewards_arm
        self.store_sorted_rewards_arm = store_sorted_rewards_arm
        self.store_all_sorted_rewards = store_all_sorted_rewards
        self.q_left = q_left
        self.q_right = q_right
        self.binarization_trick = binarization_trick
        self.current_max = -np.inf
        self.max_arms = np.zeros(self.nb_arms)
        self.reset()

    def reset(self):
        """
        Initialization of quantities of interest used for all methods
            - Sa: np.array, cumulative reward of arm a
            - Na: np.array, number of times arm a has been pulled
            - reward: np.array, rewards
            - arm_sequence: np.array, arm chose at each step
        """
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms, dtype='int')
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.zeros(self.T, dtype=int)
        self.t = 0
        if self.binarization_trick:
            self.Sa_bin = np.zeros(self.nb_arms)
            self.Na_bin = np.zeros(self.nb_arms, dtype='int')
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_max_rewards_arm:
            self.max_rewards_arm = [-np.inf for _ in range(self.nb_arms)]
        if self.store_sorted_rewards_arm:
            self.sorted_rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_all_sorted_rewards:
            self.all_sorted_rewards = []
            #self.idx_q_left = np.zeros(self.nb_arms, dtype='int')
            #self.idx_q_right = np.zeros(self.nb_arms, dtype='int')

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if self.store_max_rewards_arm:
            if reward > self.max_rewards_arm[arm]:
                self.max_rewards_arm[arm] = reward
        if self.store_sorted_rewards_arm:
            insort(self.sorted_rewards_arm[arm], reward)
        if self.store_all_sorted_rewards:
            insort(self.all_sorted_rewards, reward)
        if self.current_max < reward:
            self.current_max = reward
        if self.max_arms[arm] < reward:
            self.max_arms[arm] = reward

    def update_binarized(self, t, arm, reward_bin):
        """
        Update all the parameters of interest for the binarization trick
        after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa_bin:  np.array,  number of successes array up to time t-1
        :param Na_bin:  np.array, number of times arm has been pulled up to time t-1
        """
        self.Na_bin[arm] += 1
        self.Sa_bin[arm] += reward_bin

    def regret(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.means.max() * np.arange(1, self.T + 1) - np.cumsum(np.array(self.means)[self.arm_sequence])


class TrackerMax:
    def __init__(self,
                 nb_arms, T,
                 store_rewards_arm=False,
                 store_max_rewards_arm=False,
                 store_sorted_rewards_arm=False,
                 store_all_sorted_rewards=False,
                 store_maxima=False
                 ):
        self.nb_arms = nb_arms
        self.T = T
        self.store_rewards_arm = store_rewards_arm
        self.store_max_rewards_arm = store_max_rewards_arm
        self.store_sorted_rewards_arm = store_sorted_rewards_arm
        self.store_all_sorted_rewards = store_all_sorted_rewards
        self.store_maxima = store_maxima
        self.current_max = -np.inf
        self.max_arms = np.zeros(self.nb_arms)
        self.reset()

    def reset(self):
        """
        Initialization of quantities of interest used for all methods
            - reward: np.array, rewards
            - arm_sequence: np.array, arm chose at each step
            ...
        """
        self.Na = np.zeros(self.nb_arms, dtype='int')
        self.t = 0
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_max_rewards_arm:
            self.max_rewards_arm = [-np.inf for _ in range(self.nb_arms)]
        if self.store_sorted_rewards_arm:
            self.sorted_rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_all_sorted_rewards:
            self.all_sorted_rewards = []
        if self.store_maxima:
            self.maxima = dict(zip(np.arange(self.nb_arms), [{} for _ in range(self.nb_arms)]))
            self.n = np.zeros(self.nb_arms, dtype=np.int32)
            self.nb_batch = np.zeros(self.nb_arms, dtype=np.int32)
            self.qomax = np.inf * np.ones(self.nb_arms)

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, new_rewards
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if self.store_max_rewards_arm:
            if reward > self.max_rewards_arm[arm]:
                self.max_rewards_arm[arm] = reward
        if self.store_sorted_rewards_arm:
            insort(self.sorted_rewards_arm[arm], reward)
        if self.store_all_sorted_rewards:
            insort(self.all_sorted_rewards, reward)
        if self.current_max < reward:
            self.current_max = reward
        if self.max_arms[arm] < reward:
            self.max_arms[arm] = reward

    def efficient_update(self, dic, new_idx, new_val):
        idx, val = [*dic.key()], [*dic.values()]
        idx_val = bisect(val, new_val)
        idx = idx[:idx_val]
        val = val[:idx_val]
        idx.append(new_idx)
        val.append(new_val)
        return dict(zip(idx, val))

    def create_batch(self, arm, distrib):
        nb_current = min(int(self.T - self.t), int(self.n[arm]))
        self.t += nb_current
        self.Na[arm] += nb_current
        rewards = distrib.sample(size=nb_current)
        rwd = [(i, rewards[i]) for i in range(rewards.shape[0])]
        batch = []
        while len(rwd) > 0:
            batch = [rwd[-1]] + batch
            rwd = [x for x in rwd if x[1] > rwd[-1][1]]
        self.maxima[arm][int(self.nb_batch[arm])] = dict(batch)
        self.nb_batch[arm] += 1

    def update_batches(self, arm, distrib):
        nb_current = min(self.nb_batch[arm], self.T - self.t)
        rewards = distrib.sample(size=int(nb_current))
        self.Na[arm] += nb_current
        self.t += int(nb_current)
        for batch in range(int(nb_current)):
            self.maxima[arm][int(batch)] = {key: val for key, val in self.maxima[arm][batch].items() if
                                            val > rewards[batch]}
            self.maxima[arm][int(batch)][self.n[arm]] = rewards[batch]

    def collect_rewards_batch(self, MAB, chosen_arms, l, batch_size):
        np.random.shuffle(chosen_arms)
        for arm in chosen_arms:
            self.n[arm] += 1
            if self.nb_batch[arm] > 0:  # Update Existing batches
                self.update_batches(arm, MAB[arm])
            while batch_size(self.n[arm]) > self.nb_batch[arm] and chosen_arms != [l] and self.T - self.t > 0:  # Add a new batch to the challenger if necessary
                self.create_batch(arm, MAB[arm])
        while self.nb_batch.max() > self.nb_batch[l] and self.T - self.t > 0:  # add new batch to the leader too
            self.create_batch(l, MAB[l])

    def compute_max_from_dic(self):
        for k in range(self.nb_arms):
            for i in range(self.nb_batch[k]):
                m = list(self.maxima[k][i].values())[0]
                if m > self.current_max:
                   self.current_max = m