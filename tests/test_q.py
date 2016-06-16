from state import State
from mdp import ClassicMDP
from policy import ClassicPolicy
from gridworld import BasicGrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from analysis import Analysis
import IPython
import plot_class
from qlearner import QLearner
from policy import QPolicy
import scenarios
from svm_dagger import SVMDagger
plotter = plot_class.Plotter()

ITER = 15
TRIALS = 20
SAMP = 5

H = 15
W = 15

grid = BasicGrid(H, W)
rewards = scenarios.scenario2['rewards']
sinks = scenarios.scenario2['sinks']
grid.reward_states = rewards
grid.sink_states = sinks
mdp = ClassicMDP(ClassicPolicy(grid), grid)

#mdp.value_iteration()
#mdp.save_policy(filename='scen1.p')
mdp.load_policy(filename='scen1.p')

value_iter_pi = mdp.pi

plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states)


value_iter_data = np.zeros([TRIALS, ITER])
classic_q_data = np.zeros([TRIALS, ITER])

for t in range(TRIALS):
    mdp.load_policy(filename='scen1.p')
    q = QLearner(grid, mdp, moves=40)
    r = 0.0
    for i in range(ITER):
        q.guide()
        r = r + q.get_reward() / (ITER)
    print "Value iter reward: " + str(r)
    value_iter_data[t,:] = np.zeros(ITER) + r

    r = 0.0
    
    q.clear_states()
    mdp.pi = QPolicy(q)    
    a = Analysis(W, H, ITER, rewards=rewards, sinks=sinks, desc='Q policy')
    for i in range(ITER * SAMP):
        q.rollout()
        r = r + q.get_reward() / (ITER * SAMP)
    print "Q learn reward: " + str(r)
    if t == 0:
        a.count_states(q.get_states())
        a.show_states()
        plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states)    
    classic_q_data[t,:] = np.zeros(ITER) + r



# DAGGER

dagger_data = np.zeros((TRIALS, ITER))
dagger_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Dagger's policy progression")
for t in range(TRIALS):
    print "Trial: " + str(t)
    mdp.load_policy(filename='scen1.p')
    dagger = SVMDagger(grid, mdp)
    dagger.rollout()
    r = np.zeros(ITER)
    
    for _ in range(ITER):
        dagger.retrain()

        for i in range(SAMP):
            dagger.rollout()
            r[_] = r[_] + dagger.get_reward() / SAMP
    if t == 0:
        dagger_analysis.reset_density()        
        dagger_analysis.count_states(dagger.get_states())
        dagger_analysis.show_states()
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks)

    dagger_data[t,:] = r
    



plot_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Reward comp.")
plot_analysis.get_perf(value_iter_data)
plot_analysis.get_perf(classic_q_data)
plot_analysis.get_perf(dagger_data)

plot_analysis.plot(names = ['Value iter.', 'Guided Monte Carlo', 'DAGGER'], label='Reward')

"""
q = QLearner(grid, mdp, moves=20)
q.animate = False
for i in range(SAMP):
    q.guide()
for key in q.Q.dataset.keys():
    print key, ",", q.Q.dataset[key]

a = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Q policy')
q.clear_states()
print len(q.states)
mdp.pi = QPolicy(q)
q.animate = False

for i in range(SAMP):
    q.rollout()

a.count_states(q.get_states())
a.show_states()

q.animate=True
q.rollout()

"""
