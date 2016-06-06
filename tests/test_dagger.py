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

ITER = 100
TRIALS =1
SAMP = 20

#GridWorld Params
H = 15
W = 15

grid = BasicGrid(H, W)
mdp = ClassicMDP(ClassicPolicy(grid), grid)
analysis = Analysis(H,W,TRIALS, "DAgger video motion")
#mdp.value_iteration()
#mdp.save_policy()
mdp.load_policy()

data = np.zeros([TRIALS,ITER])
for k in range(TRIALS):
    print "Running trial", k
    mdp.load_policy()
    dagger = Dagger(grid, mdp)
    dagger.rollout()
    r_D = np.zeros(ITER)
    last_iteration_states = None
    iteration_states = []
    analysis.reset_density()
    for t in range(ITER):
        print "DAgger iteration:", t
        dagger.retrain()
        for i in range(SAMP):
            dagger.rollout()
            iteration_states += dagger.get_recent_rollout_states().tolist()
            r_D[t] = r_D[t]+dagger.get_reward()/SAMP
        analysis.count_states(np.array(iteration_states))
        analysis.save_states("images/dagger" + str(t) + ".png")

    analysis2 = Analysis(H, W, TRIALS, "DAgger final policy")
    iteration_states = []
    for i in range(SAMP):
        dagger.rollout()
        iteration_states += dagger.get_recent_rollout_states().tolist()
    analysis2.count_states(np.array(iteration_states))
    analysis2.save_states('images/final_policy.png')
    data[k,:] = r_D

