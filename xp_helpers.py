from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import pickle as pkl
from time import time
import os
from Extreme import GenericMAB as GM
pickle_path = "/xp"


def MC_Xtreme(args):
    arms, coef, algs, n_xp, T, param = args
    model = GM(arms, coef)
    res = {}
    for alg in algs:
        res[alg] = model.MC(alg, n_xp, T, param[alg])
    return res


def multiprocess_MC_Xtreme(args,
                    pickle_path=None, caption=None
                    ):
    t0 = time()
    cpu = mp.cpu_count()
    print('Running on %i clusters' % cpu)
    arms, coef, algs, n_xp, T, param = args
    new_args = (arms, coef, algs, n_xp//cpu+1,  T, param)
    res = Parallel(n_jobs=cpu)(delayed(MC_Xtreme)(new_args) for _ in range(cpu))
    m, K = res[0][algs[0]][1].shape
    param_copy = {}
    for alg in param.keys():
        param_copy[alg] = {}
        for arg in param[alg].keys():
            if callable(param[alg][arg]):
                param_copy[alg][arg] = str(param[alg][arg])
            else:
                param_copy[alg][arg] = param[alg][arg]
    all_res = {'Info': {'Arm types': arms, 'param_distrib': coef, 'N_xp': n_xp,
                        'T': T, 'algorithms': algs, 'parameters': param_copy}}
    for alg in algs:
        agg_count = np.zeros(K)
        all_counts = np.zeros((m * cpu, K))
        all_max = np.zeros(m * cpu)
        for i in range(cpu):
            agg_count += res[i][alg][0]/cpu
            all_counts[m*i:m*(i+1), :] = res[i][alg][1]
            all_max[m*i:m*(i+1)] = res[i][alg][2]
        all_res[alg] = {}
        all_res[alg]['Best Arm pulls (averaged)'] = agg_count
        all_res[alg]['Arm pull per trajectory'] = all_counts
        all_res[alg]['maxima of all trajectories'] = all_max
    if pickle_path is not None:
        pkl.dump(all_res,
            open(os.path.join(pickle_path, caption+'.pkl'), 'wb')
            )
    print('Execution time: {:.0f} seconds'.format(time()-t0))
    return all_res
