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

value_iter_pi = mdp.pi
plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states,
        filename='q_comparisons/value_iter_state_action.png')

value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])
for t in range(TRIALS):
    mdp.load_policy()
    sup = SVMSupervise(grid, mdp)
    sup.sample_policy()


