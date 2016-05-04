from sklearn import svm

class LinearSVM():

    def __init__(self, grid, mdp):
        self.mdp = mdp
        self.grid = grid
        self.data = {}
        self.svm = None

    def add_datum(self, state, action):
        tup = (state.x, state.y)
        self.data[tup] = action
        
    def fit(self):
        self.svm = svm.LinearSVC()
        X = []
        Y = []
        for key in self.data:
            X.append(list(key))
            Y.append(self.data[key])
        self.svm.fit(X, Y)
    
    def predict(self, a):
        return self.svm.predict(a)
