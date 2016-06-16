"""
    Comparison of value iteration (optimial policy), classic non-noisy supervisor and DAgger
    On a neural net
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
plotter = plot_class.Plotter()

ITER = 20
TRIALS = 50
SAMP = 30
#ITER = 3
#TRIALS = 2
#SAMP = 5


H = 15
W = 15

grid = BasicGrid(H, W)
grid.reward_states = scenarios.scenario2['rewards']
grid.sink_states = scenarios.scenario2['sinks']
mdp = ClassicMDP(ClassicPolicy(grid), grid)
#mdp.value_iteration()
#mdp.save_policy()
mdp.load_policy()

value_iter_pi = mdp.pi


# SUPERVISOR POLICY
for j in range(0):
    for i in range(30):
        grid.step(mdp)
    grid.show_recording()
    grid.reset_mdp()
    grid.clear_record_states()

plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states,
        filename='comparisons/value_iter_state_action.png')

# NAIVE IL SUPERVISOR
sup_data = np.zeros([TRIALS,ITER])
classic_il_data = np.zeros([TRIALS, ITER])
for t in range(TRIALS):
    mdp.load_policy()
    sup = Supervise(grid, mdp)
    sup.sample_policy()

    supervisor_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Supervisor's policy")
    
    r = 0.0
    for _ in range(ITER * SAMP):
        sup.rollout()
        r = r + sup.get_reward() / (ITER * SAMP)
    print "Value iter reward: " + str(r)
    if t == 0:
        supervisor_analysis.count_states(sup.get_states())
        supervisor_analysis.save_states("comparisons/value_iter.png") 
        supervisor_analysis.show_states()


    sup.train()
    classic_train, classic_test = sup.net.return_stats()
    classic_train = np.zeros((TRIALS, ITER)) + classic_train
    classic_test = np.zeros((TRIALS, ITER)) + classic_test
    sup_data[t,:] = np.zeros(ITER) + r
    
    r = 0.0
    sup.net.clear_data()
    sup.sample_policy()
    il_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks = grid.sink_states, desc="IL's policy")
    print sup.get_states()
    for _ in range(SAMP * ITER):
        sup.animate = False
        sup.rollout()
        r = r + sup.get_reward() / (SAMP * ITER)
    
    print "Classic IL reward: " + str(r)
    if t == 0:
        il_analysis.count_states(sup.get_states())
        il_analysis.save_states("comparisons/net_classic_il.png")
        il_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states,
                filename='comparisons/net_classic_il_state_action.png')
    classic_il_data[t,:] = np.zeros(ITER) + r
    
    
# DAGGER
dagger_data = np.zeros((TRIALS, ITER))
dagger_train, dagger_test = np.zeros((TRIALS, ITER)), np.zeros((TRIALS, ITER))
dagger_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Dagger's policy progression")
for t in range(TRIALS):
    mdp.load_policy()
    dagger = Dagger(grid, mdp)
    dagger.rollout()
    r_D = np.zeros(ITER)
    dagger_test_acc = np.zeros(ITER)
    dagger_train_acc = np.zeros(ITER)
    for _ in range(ITER):
        print "Dagger iteration:", _
        dagger.retrain()
        dagger_train_acc[_], dagger_test_acc[_] = dagger.net.return_stats()
        
        iteration_states = []        
        for i in range(SAMP):
            dagger.rollout()
            iteration_states += dagger.get_recent_rollout_states().tolist()            
            r_D[_] = r_D[_]+dagger.get_reward() / SAMP
        if _ == ITER - 1 and t == 0:
            dagger_analysis.count_states(np.array(iteration_states))
            dagger_analysis.save_states("comparisons/net_dagger_final.png")            
            dagger_analysis.show_states()
    if t == 0:
        dagger_analysis.reset_density()        
        dagger_analysis.count_states(dagger.get_states())
        dagger_analysis.save_states("comparisons/net_dagger.png")
        dagger_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states,
                filename='comparisons/net_dagger_state_action.png')
    dagger_train[t,:] = dagger_train_acc
    dagger_test[t,:] = dagger_test_acc
    dagger_data[t,:] = r_D


# Give outputs, save and plot rewards/accuracies
print sup_data
print classic_il_data
print dagger_data

np.save('data/net_sup_data.npy', sup_data)
np.save('data/net_classic_il_data.npy', classic_il_data)
np.save('data/net_dagger_data.npy', dagger_data)

np.save('data/net_classic_test.npy', classic_test)
np.save('data/net_classic_train.npy', classic_train)
np.save('data/net_dagger_test.npy', dagger_test)
np.save('data/net_dagger_train.npy', dagger_train)

print classic_test
print classic_train
print dagger_test
print dagger_train


analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="General comparison")
analysis.get_perf(sup_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)

analysis.plot(names = ['Value iteration', 'Classic IL', 'DAgger'], filename='comparisons/net_reward_comparison.png')

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
acc_analysis.get_perf(classic_test)
acc_analysis.get_perf(classic_train)
acc_analysis.get_perf(dagger_test)
acc_analysis.get_perf(dagger_train)

acc_analysis.plot(names = ['Classic IL test', 'Classic IL train', 'DAgger test', 'DAgger train'], label='Accuracy', filename='comparisons/net_acc_comparison.png')


