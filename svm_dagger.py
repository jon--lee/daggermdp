from svm import LinearSVM
from policy import SVMPolicy
from state import State
class SVMDagger():

    def __init__(self, grid, mdp, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.svm = LinearSVM(grid, mdp)
        self.moves = moves
        self.super_pi = mdp.pi
        
    def rollout(self):
        self.grid.reset_mdp()
        for _ in range(self.moves):
            self.svm.add_datum(self.mdp.state, self.super_pi.get_next(self.mdp.state))
            self.grid.step(self.mdp)

        self.grid.show_recording()
        #print self.svm.data


    def set_supervisor_pi(self, pi):
        self.super_pi = pi

    def retrain(self):
        self.svm.fit()
        self.mdp.pi = SVMPolicy(self.svm)
        #print self.mdp.pi.get_next(State(0,0))
