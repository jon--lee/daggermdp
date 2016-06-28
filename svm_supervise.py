from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np
import IPython
from supervise import Supervise
from state import State

class SVMSupervise(Supervise):

    def __init__(self, grid, mdp, moves=40, depth=10):
        Supervise.__init__(self, grid, mdp, moves)
        self.net = self.svm
        self.depth = depth

    def train(self):
        self.net.fit(depth=self.depth)
        self.mdp.pi = SVMPolicy(self.svm)
        self.record = False

        
