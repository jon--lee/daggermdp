from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP
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

ITER = 10
TRIALS =1
SAMP = 5

#GridWorld Params
H = 15
W = 15

grid = BasicGrid(H, W)
mdp = ClassicMDP(ClassicPolicy(grid), grid)
analysis = Analysis(H,W,TRIALS)
# mdp.value_iteration()
# mdp.save_policy()
mdp.load_policy()


#####DATA COLLECTION#####
data = np.zeros([TRIALS,ITER])
test_loss_n = np.zeros([TRIALS])
train_loss_n = np.zeros([TRIALS])
for k in range(TRIALS):
	mdp.load_policy()
	nsupervise = NSupervise(grid,mdp)
	#Collect Noisy Supervise Samples
	
	for t in range(ITER*SAMP):
	   	nsupervise.rollout()
	nsupervise.train()
	#Evaluate Policy
	r = 0.0
	for t in range(SAMP):
		nsupervise.rollout()
		r = r+nsupervise.get_reward()/SAMP
	r_SN = np.zeros(ITER)+r
	data[k,:] = r_SN

	analysis.count_states(nsupervise.get_states())
	test_loss_n[k] = nsupervise.get_test_loss()
	train_loss_n[k] = nsupervise.get_train_loss() 

analysis.show_states()

analysis = Analysis(H,W,TRIALS)

###POLICY EVAULATION###
for k in range(TRIALS):
	#Collect Noisy Supervise Samples
	
	for t in range(ITER*SAMP):
	   	nsupervise.rollout()

	analysis.count_states(nsupervise.get_states())

analysis.show_states()



