"""
    Comparison of value iteration (optimial policy), classic noisy supervisor and DAgger
    on a Linear SVM
"""


from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP
from svm_dagger import SVMDagger
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
from svm_nsupervise import SVMNSupervise
plotter = plot_class.Plotter()


ITER = 20
TRIALS = 50
SAMP = 30
#ITER = 4
#TRIALS = 20
#SAMP = 10

H = 15
W = 15

grid = BasicGrid(H, W)
rewards = scenarios.scenario2['rewards']
sinks = scenarios.scenario2['sinks']
grid.reward_states = rewards
grid.sink_states = sinks

mdp = ClassicMDP(ClassicPolicy(grid), grid)
#mdp.value_iteration()
#mdp.save_policy()
mdp.load_policy()

value_iter_pi = mdp.pi
plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states,
        filename='comparisons/noisy_comparisons/value_iter_state_action.png')

value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])
for t in range(TRIALS):
    mdp.load_policy()
    sup = SVMNSupervise(grid, mdp)
    sup.sample_policy()


    value_iter_analysis = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Value iter policy')

    r = 0.0
    for _ in range(ITER * SAMP):
        sup.rollout()
        r = r + sup.get_reward() / (ITER * SAMP)
    print "Value iter reward: " + str(r)
    if t == 0:
        value_iter_analysis.count_states(sup.get_states())
        value_iter_analysis.save_states("comparisons/noisy_comparisons/value_iter.png")
        value_iter_analysis.show_states()

    sup.train()
    value_iter_data[t,:] = np.zeros(ITER) + r

    r = 0.0
    sup.net.clear_data()
    sup.sample_policy()
    il_analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="IL's policy")    
    for _ in range(SAMP * ITER):
        sup.animate = False
        sup.rollout()
        r = r + sup.get_reward() / (SAMP * ITER)

    print "Classic IL reward: " + str(r)
    if t == 0:
        il_analysis.count_states(sup.get_states())
        il_analysis.save_states("comparisons/noisy_comparisons/noisy_classic_il.png")
        il_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename='comparisons/noisy_comparisons/noisy_classic_il_state_action.png')
    classic_il_data[t,:] = np.zeros(ITER) + r
    


# DAGGER


dagger_data = np.zeros((TRIALS, ITER))
dagger_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Dagger's policy progression")
for t in range(TRIALS):
    print "Trial: " + str(t)
    mdp.load_policy()
    dagger = SVMDagger(grid, mdp)
    #mdp.pi_noise =True
    dagger.rollout()
    #mdp.pi_noise=False
    r = np.zeros(ITER)
    for _ in range(ITER):
        dagger.retrain()
        iteration_states = []
        
        for i in range(SAMP):
            dagger.rollout()
            iteration_states += dagger.get_recent_rollout_states().tolist()            
            r[_] = r[_] + dagger.get_reward() / SAMP
        if _ == ITER - 1 and t == 0:
            dagger_analysis.count_states(np.array(iteration_states))
            dagger_analysis.save_states("comparisons/noisy_comparisons/noisy_dagger_final.png")            
            dagger_analysis.show_states()
    if t == 0:
        dagger_analysis.reset_density()        
        dagger_analysis.count_states(dagger.get_states())
        dagger_analysis.save_states("comparisons/noisy_comparisons/noisy_dagger.png")
        dagger_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename='comparisons/noisy_comparisons/noisy_dagger_state_action.png')
    dagger_data[t,:] = r

print value_iter_data
print classic_il_data
print dagger_data

np.save('comparisons/noisy_data/noisy_sup_data.npy', value_iter_data)
np.save('comparisons/noisy_data/noisy_classic_il_data.npy', classic_il_data)
np.save('comparisons/noisy_data/noisy_dagger_data.npy', dagger_data)

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)

analysis.plot(names = ['Value iteration', 'Classic IL', 'DAgger'], filename='comparisons/noisy_comparisons/noisy_reward_comparison.png')


