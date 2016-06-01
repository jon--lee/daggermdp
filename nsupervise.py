from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np
import IPython

from state import State
class NSupervise():

    def __init__(self, grid, mdp, moves=40,net = 'Net'):
        self.grid = grid
        self.mdp = mdp
        self.svm = LinearSVM(grid, mdp)
        self.net = Net(grid,mdp,net)
        self.moves = moves
        #self.reward = np.zeros(40)
        self.super_pi = mdp.pi
        self.mdp.pi_noise = True
        self.reward = np.zeros(self.moves)
        self.animate = False
        self.train_loss = 0
        self.test_loss = 0
        self.record = True
        
    def rollout(self):
        self.grid.reset_mdp()
        self.reward = np.zeros(self.moves)
        for t in range(self.moves):
            a = self.super_pi.get_next(self.mdp.state)
            print "action ",a
            if(self.record):
                self.net.add_datum(self.mdp.state, a)
            #Get current state and action
            x_t = self.mdp.state
            a_t = self.mdp.pi.get_next(x_t)

            #Take next step 
            self.grid.step(self.mdp)

            x_t_1 = self.mdp.state

            #Evaualte reward recieved 
            self.reward[t] = self.grid.reward(x_t,a_t,x_t_1)


        if(self.animate):
            self.grid.show_recording()
        
        #print self.svm.data
    def sample_policy(self):
        self.record = True
        self.net.clear_data()
    def get_states(self):
        return self.net.get_states()
    def get_reward(self):
        return np.sum(self.reward)
    def set_supervisor_pi(self, pi):
        self.super_pi = pi

    def train(self):
        self.net.fit()
        stats = self.net.return_stats()
        self.train_loss = stats[0]
        self.test_loss = stats[1]
        self.mdp.pi_noise = False
        self.mdp.pi = NetPolicy(self.net)
        self.record = False

    def get_train_loss(self):
        return self.train_loss

    def get_test_loss(self):
        return self.test_loss

        #print self.mdp.pi.get_next(State(0,0))
