from svm import LinearSVM
from policy import SVMPolicy
from state import State
import numpy as np
from dagger import Dagger


class SVMDagger(Dagger):

    def __init__(self, grid, mdp, moves=40):
        Dagger.__init__(self, grid, mdp, moves)
        self.net = self.svm

    def retrain(self):
        self.net.fit()
        self.mdp.pi = SVMPolicy(self.svm)
"""
    def __init__(self, grid, mdp, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.svm = LinearSVM(grid, mdp)
        self.moves = moves
        self.super_pi = mdp.pi
        self.reward = np.zeros(self.moves)
        self.animate = False
        self.recent_rollout_states = None
        
    def rollout(self):
        self.grid.reset_mdp()
        self.recent_rollout_states = [self.mdp.state]
        for t in range(self.moves):
            self.svm.add_datum(self.mdp.state, self.super_pi.get_next(self.mdp.state))
            self.grid.step(self.mdp)
            
            x_t = self.mdp.state
            a_t = self.mdp.pi.get_next(x_t)

            self.grid.step(self.mdp)

            x_t_1 = self.mdp.state

            self.reward[t] = self.grid.reward(x_t, a_t, x_t_1)
        
            self.recent_rollout_states.append(self.mdp.state)

        if self.animate:
            self.grid.show_recording()
            
        return self.recent_rollout_states

    def set_supervisor_pi(self, pi):
        self.super_pi = pi

    def get_states(self):
        return self.svm.get_states()

    def get_reward(self):
        return np.sum(self.reward)

    def retrain(self):
        self.svm.fit()
        self.mdp.pi = SVMPolicy(self.svm)
        #print self.mdp.pi.get_next(State(0,0))

""" 
