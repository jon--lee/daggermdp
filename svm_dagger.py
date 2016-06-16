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


