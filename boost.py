from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np

class Boost():

    def __init__(self, grid, mdp):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.svm = None
        self.nonlinear = False

    def add_datum(self, state, action):
        self.data.append((state, action))

    def fit(self):
        if self.nonlinear:
            svc = svm.SVC(kernel='rbf', gamma=0.1, C=1.0)
        else:
            #svc = svm.LinearSVC()
            svc = DecisionTreeClassifier(max_depth=3)
        self.boost = AdaBoostClassifier(base_estimator=svc, n_estimators=10, algorithm='SAMME')
        
        X = []
        Y = []
        for state, action in self.data:
            X.append([state.x, state.y])
            Y.append(action)
        self.boost.fit(X, Y)

    def predict(self, a):
        return self.boost.predict(a)

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
    
