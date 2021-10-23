import numpy as np
from xp_helpers import multiprocess_MC_Xtreme

path = "xp"
m = 100  # Number of trajectories in each simulation! m=10000 in the paper's experiments

params = {'MoMax_SDA': {'fe': lambda x: max(5, np.sqrt(x)), 'batch_size': lambda n: np.sqrt(n)},
          'ExtremeETC': {'b': 1}, 'ExtremeHunter': {'steps': 1000, 'D': 1e-4, 'E': 1e-4, 'b': 1},
          'MaxMedian': {'explo_func': lambda t: 1/t}, 'Uniform_Exploration': {},
          'Threshold_Ascent': {'s': 100, 'delta': 0.1},
          'QoMax_SDA': {'fe': lambda r: max(5, np.log(r)**(3/2)), 'batch_size': lambda n: n**(2/3), 'q': 0.9},
          'QoMax_ETC': {'size': (100, 10), 'q': 0.9}
}

# Reproducing experiments for previous papers
xp1 = {'arms': ['Par'] * 5, 'coef': [[2.1, 1], [2.3, 1], [1.3, 1], [1.1, 1], [1.9, 1]]}  # xp 1 from Bhatt et al.
xp2 = {'arms': ['Par'] * 7, 'coef': [[2.5, 1], [2.8, 1], [4, 1], [3, 1], [1.4, 1], [1.9, 1], [1.4, 1.1]]}  # xp 2 from Bhatt et al.
xp3 = {'arms': ['Exp'] * 10, 'coef': [2.1, 2.4, 1.9, 1.3, 1.1, 2.9, 1.5, 2.2, 2.6, 1.4]}  # xp 3 from Bhatt et al.
xp4 = {'arms': ['G'] * 20, 'coef': [[1, sig] for sig in [1.64, 2.29, 1.79, 2.67, 1.70, 1.36, 1.90, 2.19,
                                                         0.80, 0.12, 1.65, 1.19, 1.88, 0.89, 3.35, 1.5, 2.22, 3.03, 1.08, 0.48]]}  # xp 3 from Bhatt et al.
xp5 = {'arms': ['Par'] * 3, 'coef': [[5, 1], [1.1, 1], [2, 1]]}  # xp 1 from Carpentier et Valko
xp6 = {'arms': ['Par'] * 2 + ['Mixture'], 'coef': [[1.5, 1], [3, 1], [('Dirac', 'Par'), (0.8, 0.2), (0, [1.1, 1])]]}  # xp 2 from Carpentier & Valko
xp7 = {'arms': ['LG'] * 5, 'coef': [[1, 4], [1.5, 3], [2, 2], [3, 1], [3.5, 0.5]]}
xp8 = {'arms': ['GenNorm'] * 8, 'coef': [0.2 * i for i in range(1, 9)]}

names_xp = ['xp'+str(i+1)+'_' for i in range(8)]  # change both names_xp and xp_list if running less xp
xp_list = [xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8]
algs = ['QoMax_ETC', 'QoMax_SDA', 'ExtremeETC', 'ExtremeHunter',
        'MaxMedian', 'Threshold_Ascent']  # Selection of algorithms to test
# T_list = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 50000]  # Selection of times from the paper
T_list = [5000, 10000, 15000]  # Sub-selection of times for testing

if __name__ == '__main__':
    for i, xp in enumerate(xp_list):
        for T in T_list:
            print(T)
            batch_size, sample_size = int(np.log(T)**2) + 1, int(np.log(T)) + 1
            args = (xp['arms'], xp['coef'], algs, m, T, params)
            params['QoMax_ETC']['size'] = (batch_size, sample_size)
            res = multiprocess_MC_Xtreme(args, pickle_path=path, caption=names_xp[i]+str(T))
        print('_________________________________________________________________')
