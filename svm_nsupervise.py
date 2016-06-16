from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np
import IPython
from nsupervise import NSupervise
from state import State

class SVMNSupervise(NSupervise):
    def __init__(self, grid, mdp, moves=40, net='Net'):
        NSupervise.__init__(self, grid, mdp, moves, net)
        self.net = self.svm
        self.mdp.pi_noise = True

    def train(self):
        self.net.fit()
        self.mdp.pi_noise = False
        self.mdp.pi = SVMPolicy(self.svm)
        self.record=False

