# Code for the paper *Efficient Algorithms for Extreme Bandits*

In this repository we provide the code to reproduce the experiments from the paper. 

## Reproduce experiments: run *__ main __.py*

This file provides the script that can run to reproduce the experiments of the paper. It contains several variables that define the Extreme Bandit 
problems and allow to run the code:
* *params* contains the parameters of the bandit algorithms.
* *xp1-8* contains the parameters for experiments 1-8 in the paper: a code the type of each arm, and the associated parameters
* The number of trajectories to sample *m*, the time horizons considered *T_list* and the selection of algorithms to test.

Running the scripts will run experiments 1 to 8 for the parameters considered, using the function *multiprocess_MC_Xtreme* (parallel computing using all available cores)
and store the results in a pickle file at the path specified by the variable of the same name (the directory has to exist).

## Code structure

* *Extreme.py* contains the implementation of the **Extreme Bandit algorithms** we implemented for this paper. They are encapsulated in a **MAB** object 
along with the frozen distributions representing the arms. 
* *arms.py* provides all the distributions we use in this paper in a unified way (as we use both numpy and scipy distribution) and with proper seeding.
* *tracker.py* contains the **TrackerMax** object, which stores all the functions to collect data and update indicators for each algorithm. Such object 
is defined at the beginning of each run of a bandit algorithm.
* *utils.py* contains some functions used by the algorithms, notably the functions to compute the **UCBs** of ThresholdAscent and ExtremeHunter. 
* *xp_helpers* provides the **MC_Xtreme** and **multiprocess_MC_Xtreme** that run bandit algorithms for a large number of trajectories. The second uses parallel computing.
