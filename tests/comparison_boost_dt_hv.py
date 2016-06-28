"""
    Boosting
    Comparison of value iteration (optimial policy), boosting and svm supervised learning 
    on a
"""

from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP, HighVarMDP
from svm_dagger import SVMDagger
from boost_supervise import BoostSupervise 
from gridworld import BasicGrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from dagger import Dagger
from supervise import Supervise
from nsupervise import NSupervise
from analysis import Analysis
import IPython
import plot_class
import scenarios
from svm_supervise import SVMSupervise
from scikit_supervise import ScikitSupervise
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

plotter = plot_class.Plotter()

ITER = 20
TRIALS = 50
SAMP = 30
#ITER = 2
#TRIALS = 2
#SAMP = 2
LIMIT_DATA = 1

H = 15
W = 15

grid = BasicGrid(H, W)
rewards = scenarios.scenario2['rewards']
sinks = scenarios.scenario2['sinks']
grid.reward_states = rewards
grid.sink_states = sinks

mdp = HighVarMDP(ClassicPolicy(grid), grid)
mdp.value_iteration()
mdp.save_policy('highvar.p')
mdp.load_policy('highvar.p')

value_iter_pi = mdp.pi
plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states,
        filename='comparisons/boost_hv_dt_comparisons/value_iter_state_action.png')

value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])
classic_il_acc = np.zeros([TRIALS, ITER])


for t in range(TRIALS):
    print "\nIL Trial: " + str(t)
    mdp.load_policy('highvar.p')

    dt = DecisionTreeClassifier(max_depth=3)
    boost = AdaBoostClassifier(base_estimator=dt, n_estimators=10, algorithm='SAMME')
    sup = ScikitSupervise(grid, mdp, Classifier=boost)
    #sup = BoostSupervise(grid, mdp)
    
    #sup.boost.nonlinear = False

    sup.sample_policy()

    value_iter_analysis = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Value iter policy')


    value_iter_r = np.zeros(ITER)
    classic_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)

    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi 
        sup.record = True
        for _ in range(SAMP):
            if _  >= LIMIT_DATA:
                sup.record = False
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / (SAMP)

        sup.record = False
        print "     Training on " + str(len(sup.net.data)) + " examples"
        sup.train()

        acc[i] = sup.svm.acc()
        for _ in range(SAMP):
            sup.rollout()
            classic_il_r[i] += sup.get_reward() / SAMP

    if t == 0:
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename='comparisons/boost_hv_dt_comparisons/boost_hv_dt_classic_il_state_action.png')        

    classic_il_data[t,:] = classic_il_r
    value_iter_data[t,:] = value_iter_r
    classic_il_acc[t,:] = acc





#DAGGER

dagger_data = np.zeros((TRIALS, ITER))
dagger_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Dagger's policy progression")
dagger_acc = np.zeros((TRIALS, ITER))
for t in range(TRIALS):
    print "DAgger Trial: " + str(t)
    mdp.load_policy('highvar.p')
    dagger = SVMDagger(grid, mdp)
    dagger.svm.nonlinear=False
    dagger.rollout()
    r = np.zeros(ITER)
    acc = np.zeros(ITER)
    for _ in range(ITER):
        print "     Iteration: " + str(_)
        print "     Retraining with " + str(len(dagger.net.data)) + " examples"
        dagger.retrain()
        acc[_] = dagger.svm.acc()
        iteration_states = []
        dagger.record = True
        for i in range(SAMP):
            if i >= LIMIT_DATA:
                dagger.record = False
            dagger.rollout()
            iteration_states += dagger.get_recent_rollout_states().tolist()            
            r[_] = r[_] + dagger.get_reward() / SAMP
        if _ == ITER - 1 and t == 0:
            dagger_analysis.count_states(np.array(iteration_states))
            dagger_analysis.save_states("comparisons/boost_hv_dt_comparisons/boost_hv_dt_dagger_final.png")            
            dagger_analysis.show_states()
    if t == 0:
        dagger_analysis.reset_density()        
        dagger_analysis.count_states(dagger.get_states())
        dagger_analysis.save_states("comparisons/boost_hv_dt_comparisons/boost_hv_dt_dagger.png")
        dagger_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename='comparisons/boost_hv_dt_comparisons/boost_hv_dt_dagger_state_action.png')
    dagger_data[t,:] = r
    dagger_acc[t,:] = acc


# print value_iter_data
# print classic_il_data
# print dagger_data



np.save('comparisons/boost_hv_dt_data/boost_hv_dt_sup_data.npy', value_iter_data)
np.save('comparisons/boost_hv_dt_data/boost_hv_dt_classic_il_data.npy', classic_il_data)
np.save('comparisons/boost_hv_dt_data/boost_hv_dt_dagger_data.npy', dagger_data)

np.save('comparisons/boost_hv_dt_data/boost_hv_dt_dagger_acc.npy', dagger_acc)
np.save('comparisons/boost_hv_dt_data/boost_hv_dt_classic_il_acc.npy', classic_il_acc)

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)

analysis.plot(names = ['HV Value iteration', 'HV AdaBoost IL', 'HV DT DAgger'], filename='comparisons/boost_hv_dt_comparisons/boost_hv_dt_reward_comparison.png', ylims=[-60, 100])

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
acc_analysis.get_perf(classic_il_acc)
acc_analysis.get_perf(dagger_acc)

acc_analysis.plot(names = ['HV AdaBoost IL Acc.', 'HV DT DAgger Acc.'], label='Accuracy', filename='comparisons/boost_hv_dt_comparisons/boost_hv_dt_acc_comparison.png', ylims=[0,1])














