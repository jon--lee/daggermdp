from svm import LinearSVM
from policy import SVMPolicy
from state import State
import numpy as np
from dagger import Dagger


class SVMDagger(Dagger):

    def __init__(self, grid, mdp, moves=40, depth=10):
        Dagger.__init__(self, grid, mdp, moves)
        self.net = self.svm
        self.depth = depth

    def retrain(self):
        self.net.fit(depth=self.depth)
        self.mdp.pi = SVMPolicy(self.svm)

