from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP
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
plotter = plot_class.Plotter()

ITER = 20
TRIALS = 50
SAMP = 30


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

sup = BoostSupervise(grid, mdp)
sup.boost.nonlinear = False
sup.sample_policy()
sup.animate = False
sup.record = True
T = 2
K = 1
avg = 0.0
sup_rewards = []
for i in range(T):
    if i >= K:

        sup.record = False
    sup.rollout()
    sup_rewards.append(sup.get_reward())
    avg += sup.get_reward() / float(T)
    #print "Supervisor reward: "  + str(sup.get_reward())
print "Supervisor Avg: " + str(avg)
print sup_rewards

sup.record = False
sup.train()
plotter.plot_state_actions(mdp.pi, rewards = rewards, sinks=sinks)

avg = 0.0
rol_rewards = []
# print "data: " + str(np.array(sup.boost.data).shape)
# print "data: " + str(sup.boost.data)
for t in range(T):
    sup.rollout()
    rol_rewards.append(sup.get_reward())
    avg += sup.get_reward() / float(T)
    #print "Rollout reward: " + str(sup.get_reward())
print "Rollout Avg: " + str(avg)
print rol_rewards



