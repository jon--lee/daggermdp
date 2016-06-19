from boost import Boost
from policy import BoostPolicy
from supervise import Supervise
class BoostSupervise(Supervise):

    def __init__(self, grid, mdp, moves=40):
        Supervise.__init__(self, grid, mdp, moves)
        self.boost = Boost(grid, mdp)
        self.svm = self.boost
        self.net = self.boost
        

    def train (self):
        self.boost.fit()
        self.mdp.pi = BoostPolicy(self.boost)
        self.record = False
