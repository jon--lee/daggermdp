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
TRIALS =20
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

####DAgger##########
data = np.zeros([TRIALS,ITER])
for k in range(TRIALS):
	mdp.load_policy()
	dagger = Dagger(grid, mdp)
	dagger.rollout()            # rollout with supervisor policy
	r_D = np.zeros(ITER)
	for t in range(ITER):
		dagger.retrain()
		for i in range(SAMP):
			dagger.rollout()
			r_D[t] = r_D[t]+dagger.get_reward()/SAMP

	data[k,:] = r_D
	#analysis.show_states(dagger.get_states())

analysis.get_perf(data)

####SUPERVISE########
data = np.zeros([TRIALS,ITER])
for k in range(TRIALS):
	mdp.load_policy()
	supervise = Supervise(grid,mdp)

	#Collect Supervise Samples
	for t in range(ITER*SAMP):
	   	supervise.rollout()
	supervise.train()
	#Evaluate Policy
	r = 0.0
	for t in range(SAMP):
		supervise.rollout()
		r = r+supervise.get_reward()/SAMP
	r_S = np.zeros(ITER)+r
	data[k,:] = r_S
	#analysis.show_states(supervise.get_states())

analysis.get_perf(data)


#####NOISY SUPERVISOR#####
# data = np.zeros([TRIALS,ITER])
# for k in range(TRIALS):
# 	mdp.load_policy()
# 	nsupervise = NSupervise(grid,mdp)
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

#analysis.get_perf(data)
analysis.plot()

# IPython.embed()
# plt.plot(r_S)
# plt.plot(r_SN)

# plt.show()