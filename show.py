from analysis import Analysis
import numpy as np

if __name__ == '__main__':
    value_iter_data = np.load('comparisons/boost_hv_dt_data/boost_hv_dt_sup_data.npy')
    classic_il_data = np.load('comparisons/boost_hv_dt_data/boost_hv_dt_classic_il_data.npy')
    dagger_data = np.load('comparisons/boost_hv_dt_data/boost_hv_dt_dagger_data.npy')

    analysis = Analysis(15, 15, 19, rewards = None, sinks=None)
    analysis.get_perf(value_iter_data)
    analysis.get_perf(classic_il_data)
    analysis.get_perf(dagger_data)
    analysis.plot(names = ['HV Value iteration', 'HV AdaBoost IL', 'HV DT DAgger'], filename='tset.png', ylims=[-20, 15])
    
    
