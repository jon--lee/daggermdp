from sklearn import svm

class LinearSVM():

    def __init__(self, grid, mdp):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.svm = None

    def add_datum(self, state, action):
        self.data.append((state, action))
        
    def fit(self):
        self.svm = svm.LinearSVC()
        X = []
        Y = []
        for state, action in self.data:
            X.append([state.x, state.y])
            Y.append(action)
        self.svm.fit(X, Y)
    
    def predict(self, a):
        return self.svm.predict(a)
