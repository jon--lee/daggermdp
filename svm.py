from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import Perceptron
import numpy as np
class LinearSVM():

    def __init__(self, grid, mdp):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.svm = None
        self.nonlinear = False
        self.perceptron = False
        
    def add_datum(self, state, action):
        self.data.append((state, action))
        
    def fit(self, depth=10):
        if self.nonlinear:
            print "     RBF SVM"
            self.svm = svm.SVC(kernel='rbf', gamma=0.1, C=1.0)
        else:
            print "     Linear SVM"
            self.svm = svm.SVC(kernel='linear')
        if self.perceptron:
            print "     perceptron"
            self.svm = Perceptron()
        X = []
        Y = []
        for state, action in self.data:
            X.append([state.x, state.y])
            Y.append(action)
        self.svm.fit(X, Y)
    
    def predict(self, a):
        return self.svm.predict(a)

    def get_states(self):
        N = len(self.data)
        states = np.zeros([N,2])
        for i in range(N):
            x = self.data[i][0].toArray()
            states[i,:] = x

        return states


    def clear_data(self):
        self.data = []


    def acc(self):
        results = []
        for s, a in self.data:
            pred = self.predict([[s.x, s.y]])[0]
            results.append(pred == a)
        return float(sum(results)) / float(len(self.data))
