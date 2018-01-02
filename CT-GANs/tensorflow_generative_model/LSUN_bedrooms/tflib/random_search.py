import itertools
import numpy as np

def random_search(configs, n_trials=-1, n_splits=1, split=0):
    keys = [key for key, vals in configs]
    vals = [vals for key, vals in configs]
    all_trials = [x for x in itertools.product(*vals)]
    random_state = np.random.RandomState(42)
    random_state.shuffle(all_trials)
    if n_trials != -1:
        all_trials = all_trials[:n_trials]
    for trial in all_trials[split::n_splits]:
        config_dict = {k:v for k,v in zip(keys, trial)}
        yield config_dict