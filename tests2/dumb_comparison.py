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
from scikit_supervise import ScikitSupervise
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifierCV, Perceptron, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import os
plotter = plot_class.Plotter()


ITER = 10
TRIALS = 30
SAMP = 20

LIMIT_DATA = 1
DEPTH = 10

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

dumb_policy = DumbPolicy(grid)
value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])


plotter.plot_state_actions(dumb_policy, rewards = grid.reward_states, sinks = grid.sink_states)
plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states)


for t in range(TRIALS):
    print "\nIL Trial: " + str(t)
    mdp.load_policy()

    sup = ScikitSupervise(grid, mdp, Classifier=LinearSVC())
    sup.sample_policy()
    
    value_iter_r = np.zeros(ITER)
    classic_il_r = np.zeros(ITER)

    sup.record=False
    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi 
        for _ in range(SAMP):
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / (SAMP)
        sup.record = False
        sup.mdp.pi = dumb_policy
        mdp.pi = dumb_policy
        for _ in range(SAMP):
            sup.record=False
            sup.rollout()
            classic_il_r[i] += sup.get_reward() / SAMP


    classic_il_data[t,:] = classic_il_r
    value_iter_data[t,:] = value_iter_r

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)

analysis.plot(names = ['Value iteration','Dumb Policy'], ylims=[-60, 100])

