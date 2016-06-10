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

ITER = 1
TRIALS =40
SAMP = 10

#GridWorld Params
H = 15
W = 15

grid = BasicGrid(H, W)
mdp = ClassicMDP(ClassicPolicy(grid), grid)
analysis = Analysis(H,W,TRIALS)
# mdp.value_iteration()
# mdp.save_policy()
mdp.load_policy()
analysis = Analysis(H,W,TRIALS)

####SUPERVISOR#####
# data = np.zeros([TRIALS,ITER])
# test_loss_n = np.zeros([TRIALS])
# train_loss_n = np.zeros([TRIALS])
# for k in range(TRIALS):
# 	mdp.load_policy()
# 	nsupervise = NSupervise(grid,mdp)# net = 'UB')
# 	#Collect Noisy Supervise Samples
	
# 	for t in range(ITER*SAMP):
# 	   	nsupervise.rollout()
# 	nsupervise.train()
# 	#Evaluate Policy
# 	r = 0.0
	
# 	for t in range(SAMP):
# 	 	nsupervise.rollout()
# 		r = r+nsupervise.get_reward()/SAMP
# 		r_SN = np.zeros(ITER)+r
# 	 	data[k,:] = r_SN

# analysis.get_perf(data)

	
	

# #####UB SUPERVISOR#####
data = np.zeros([TRIALS,ITER])
test_loss_n = np.zeros([TRIALS])
train_loss_n = np.zeros([TRIALS])
for k in range(TRIALS):
	mdp.load_policy()
	nsupervise = NSupervise(grid,mdp)# net = 'UB')
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

analysis.get_perf(data)


analysis.plot()
	

		
	
