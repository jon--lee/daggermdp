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


#####NOISY SUPERVISOR#####
data = np.zeros([TRIALS,ITER])
test_loss_n = np.zeros([TRIALS])
train_loss_n = np.zeros([TRIALS])
for k in range(TRIALS):
	mdp.load_policy()
	nsupervise = Supervise(grid,mdp)
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
analysis.get_perf(data)




# #####NOISY SUPERVISOR LOGISTIC#####
# data = np.zeros([TRIALS,ITER])
# test_loss = np.zeros([TRIALS])
# train_loss = np.zeros([TRIALS])
# for k in range(TRIALS):
# 	mdp.load_policy()
# 	nsupervise = NSupervise(grid,mdp, net = 'Log')
# 	#Collect Noisy Supervise Samples
	
# 	for t in range(ITER*SAMP):
# 	   	nsupervise.rollout()
# 	nsupervise.train()
# 	#Evaluate Policy
# 	r = 0.0
# 	for t in range(SAMP):
# 		nsupervise.rollout()
# 		r = r+nsupervise.get_reward()/SAMP
# 	r_SN = np.zeros(ITER)+r
# 	data[k,:] = r_SN
# 	test_loss[k] = nsupervise.get_test_loss()
# 	train_loss[k] = nsupervise.get_train_loss() 


# analysis.get_perf(data)
# print "TEST LOSS ", np.sum(test_loss)/TRIALS
# print "TRAIN LOSS", np.sum(train_loss)/TRIALS
# print "TEST LOSS NET", np.sum(test_loss_n)/TRIALS
# print "TRAIN LOSS NET", np.sum(train_loss_n)/TRIALS
# analysis.plot()

# IPython.embed()
# plt.plot(r_S)
# plt.plot(r_SN)

# plt.show()