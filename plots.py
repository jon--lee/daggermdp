from analysis import Analysis
import numpy as np

a = Analysis()


value_iter_data = np.load('boost_data/boost_sup_data.npy')
classic_il_data = np.load('boost_data/boost_classic_il_data.npy')
dagger_data = np.load('boost_data/boost_dagger_data.npy')

print value_iter_data.shape

analysis = Analysis(15, 15, 20, rewards=None, sinks=None, desc="General reward comparison")

analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)

analysis.plot(names = ['Value iteration', 'Classic IL', 'DAgger'])
