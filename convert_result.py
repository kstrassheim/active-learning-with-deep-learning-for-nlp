ex = None
import pickle
with open('result.pickle', 'rb') as handle:
    ex = pickle.load(handle)

from experiments import flatten_experiments_results


res_arr = flatten_experiments_results(ex)
print(res_arr)