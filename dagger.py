from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np

from state import State
class Dagger():

    def __init__(self, grid, mdp, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.svm = LinearSVM(grid, mdp)
        self.net = Net(grid,mdp)
        self.moves = moves
        #self.reward = np.zeros(40)
        self.super_pi = mdp.pi
        self.reward = np.zeros(self.moves)
        self.animate = False
        self.record = True
        self.recent_rollout_states = None
        
    def rollout(self):
        self.grid.reset_mdp()
        self.reward = np.zeros(self.moves)
        self.recent_rollout_states = [self.mdp.state]
        for t in range(self.moves):
            if self.record:
                self.net.add_datum(self.mdp.state, self.super_pi.get_next(self.mdp.state))
            #Get current state and action
            x_t = self.mdp.state
            a_t = self.mdp.pi.get_next(x_t)

            #Take next step 
            self.grid.step(self.mdp)

            x_t_1 = self.mdp.state

            #Evaualte reward recieved 
            self.reward[t] = self.grid.reward(x_t,a_t,x_t_1)
            self.recent_rollout_states.append(self.mdp.state)

        if(self.animate):
            self.grid.show_recording()
        #print self.svm.data

    def get_states(self):
        return self.net.get_states()
    def get_reward(self):
        return np.sum(self.reward)
    def set_supervisor_pi(self, pi):
        self.super_pi = pi
    
    def get_recent_rollout_states(self):
        N = len(self.recent_rollout_states)
        states = np.zeros([N,2])
        for i in range(N):
            x = self.recent_rollout_states[i].toArray()
            states[i,:] = x        
        return states
    
    def retrain(self):
        self.net.fit()
        self.mdp.pi = NetPolicy(self.net)
        #print self.mdp.pi.get_next(State(0,0))
