from sklearn.svm import SVC
import numpy as np



class Adaboost():
        
    def __init__(self, estimator, n_estimators=5, learning_rate=1.0):
        self.lr = learning_rate
        self.estimator = estimator
        self.T = n_estimators

    def fit(self, X, y):
        self.classes = set(y)
        K = len(self.classes)
        M = len(X)
        D = np.zeros((self.T + 1, M))
        D[0] = np.ones(len(X)) / float(M) # set to uniform distr at t = 0
        h = [None] * self.T
        eps = np.zeros(self.T)
        alpha = np.zeros(self.T)
        for t in range(self.T):
            h[t] = self.estimator.fit(X, y)
            predictions = np.array(h[t].predict(X))
            
            errors = 1 - np.array(predictions == y)
            distr_t = D[t]
            eps[t] = sum(errors * np.array(distr_t)) / sum(distr_t)
            
            alpha[t] = np.log((1 - eps[t]) / eps[t]) + np.log(K - 1)
            D[t+1, :] = self.normalize(distr_t * self.compute_exp(alpha[t], errors))
        
        self.h = h
        self.alpha = alpha
        return self

    def normalize(self, distr):
        return distr / sum(distr)

    def compute_eps(self, predictions, y, distr):
        eps = 0.0
        booleans = np.abs(np.array(predictions == y) - 1)
        return sum(booleans * np.array(distr)) / sum(distr)
        
    def compute_exp(self, alpha, errors,):
        result = alpha * errors
        return np.exp(result)

    def predict(self, X):
        return np.array(self.argmax(X))

    def argmax(self, X):
        print "argmaxing"
        print len(X)
        cs = []
        for x in X:
            max_score = None
            max_c = None
            for c in self.classes:
                score = 0.0
                for t in range(self.T):
                    predictions = np.array(self.h[t].predict([x]))
                    corrects = np.array(predictions == c) + 0
                    score += self.alpha[t] * sum(corrects)
                if max_score is None or score > max_score:
                    max_c = c
                    max_score = score
            cs.append(max_c)
        return cs


if __name__ == '__main__':
    X = [[1, 2], [1,2], [4, 3], [5, 3],[ 3,2323], [1, 33]]
    y = [-123123, 434, 1, 100, 1, 1203123123,]
    svm = SVC(kernel='linear')
    ada = Adaboost(svm)
    ada.fit(X, y)
