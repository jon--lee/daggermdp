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
from qlearner import QLearner, Qapprox
from policy import QPolicy
import scenarios
from svm_dagger import SVMDagger
plotter = plot_class.Plotter()


ITER = 5
TRIALS = 20
SAMP = 5

H = 15
W = 15

grid = BasicGrid(H, W)
rewards = scenarios.scenario0['rewards']
sinks = scenarios.scenario0['sinks']
grid.reward_states = rewards
grid.sink_states = sinks
mdp = ClassicMDP(ClassicPolicy(grid), grid)

#mdp.value_iteration()
#mdp.save_policy(filename='scen1.p')
mdp.load_policy(filename='scen1.p')

value_iter_pi = mdp.pi

plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states)


q = QLearner(grid, mdp, moves=20)
q.Q = Qapprox(H, W)
q.animate = False
for i in range(20):
    q.guide()
#for key in q.Q.dataset.keys():
#    print key, ",", np.mean(q.Q.dataset[key])


an = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Q policy')
q.clear_states()
q.retrain()
mdp.pi = QPolicy(q)
#print q.Q.get(State(2, 12), -1)
#print len(q.states)
#q.animate = True

plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states)


for j in range(5):
    q.clear_states()
    for i in range(50):
        q.rollout()
        q.guide()
    q.retrain()

plotter.plot_state_actions(mdp.pi, rewards = grid.reward_states, sinks = grid.sink_states)
q.rollout()
a = mdp.pi.get_next(State(0, 0))
print "action: " + str(a)
tup = q.Q.preprocess(0, 0, a)
print q.Q.dataset[tup]
print "Actual: " + str(np.mean(q.Q.dataset[tup]))
print "predicted: " + str(q.Q.get(State(0, 0), a))

for ac in mdp.pi.available_actions:
    if ac != a:
        print "Seeing for action: " + str(ac)
        tup = q.Q.preprocess(0, 0, ac)
        if tup in q.Q.dataset:
            print "Actual: " + str(np.mean(q.Q.dataset[tup]))
            #print np.mean(q.Q.dataset[tup])            
        else:
            print "No actual"
        print "predicted: " + str(q.Q.get(State(0, 0), ac))
    
#q.animate=True
#q.rollout()

an.count_states(q.get_states())
an.show_states()




