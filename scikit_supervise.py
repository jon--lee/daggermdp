from supervise import Supervise
from sklearn import svm
from scikit_est import SKEst
from policy import SKPolicy
class ScikitSupervise(Supervise):

    def __init__(self, grid, mdp, Classifier=svm.LinearSVC(), moves=40):
        Supervise.__init__(self, grid, mdp, moves)
        self.est = SKEst(grid, mdp, Classifier)
        self.net = self.est
        self.svm = self.est

    def train(self):
        self.net.fit()
        self.mdp.pi = SKPolicy(self.svm)
        self.record = False
