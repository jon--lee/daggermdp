from sklearn.svm import SVC
import numpy as np
from sklearn.tree import DecisionTreeClassifier



class SVC_boosted():
    def __init__(self):
        self.learner_weights = []
        self.learners = []
        self.iter_max = 100
        self.delta_stop = 1e-5
        self.phi = .5

    def fit(self, X, labels):
        '''
        performs the adaboost code using classification
        '''
        # features = np.array([point[0] for point in data])
        # labels = np.array([labels[1] for point in l])
        weak_learners = []
        learner_weights = []
        data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
        i = 0

        if len(X) > 720:
            pass

        while i < 5:
            #learner = SVC(kernel='rbf', gamma=1e-1, C=1.0)#(1e-2 * (10**i))) # instantiate a new weak learner
            #learner = SVC(kernel='linear')
            learner = DecisionTreeClassifier(max_depth=3)
            learner.fit(X, labels, sample_weight=(data_weights/.025))
            err = learner.predict(X) 
            # results = learner.accuracy(data_embedding) # results is expected to an array of classes
            raw_success = np.array(err) == np.array(labels) #np.equal(err, labels)
            acc = np.sum(raw_success * data_weights)
            err = np.sum((1-raw_success)  * data_weights)

            # get the weight of the learner
            learner_weight = np.log2(self.phi*(1-err)/(err*(1-self.phi)))
            learner_weights.append(learner_weight)

            data_weights = data_weights*np.exp(-learner_weight*raw_success)
            normalization = np.sum(data_weights)
            data_weights = data_weights/normalization

            #if err >= self.phi - self.delta_stop or err == 0:
            if err == 0:
                print i
                print err
                break
            i += 1
            weak_learners.append(learner)
        self.learners = weak_learners
        self.learner_weights = learner_weights/np.sum(np.array(learner_weights))
        return self

    def predict(self, X):
        predictions = np.zeros(len(X))
        for learner, weight in zip(self.learners, self.learner_weights):
            predictions += learner.predict(X) * weight
        prediction_values = np.around(predictions)
        prediction_values[prediction_values > 3] = 3
        prediction_values[prediction_values < -1] = -1
        return prediction_values


class WeightedSamplingSVC(SVC_boosted):

    
    def fit(self, X, labels):
        '''
        performs the adaboost code using classification
        '''
        # features = np.array([point[0] for point in data])
        # labels = np.array([labels[1] for point in l])
        weak_learners = []
        learner_weights = []
        data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
        i = 0

        while i < 2:
            #learner = SVC(kernel='rbf', gamma=1e-1, C=1.0)#(1e-2 * (10**i))) # instantiate a new weak learner
            learner = SVC(kernel='linear')
            #learner = DecisionTreeClassifier(max_depth=3)
            sampledX, sampledY = self.sample(X, labels, data_weights)
            learner.fit(sampledX, sampledY)
            #learner.fit(X, labels, sample_weight=(data_weights/.025))
            err = learner.predict(X) 
            # results = learner.accuracy(data_embedding) # results is expected to an array of classes
            raw_success = np.array(err) == np.array(labels) #np.equal(err, labels)
            acc = np.sum(raw_success * data_weights)
            err = np.sum((1-raw_success)  * data_weights)

            # get the weight of the learner
            learner_weight = np.log2(self.phi*(1-err)/(err*(1-self.phi)))
            learner_weights.append(learner_weight)

            data_weights = data_weights*np.exp(-learner_weight*raw_success)
            normalization = np.sum(data_weights)
            data_weights = data_weights/normalization

            #if err >= self.phi - self.delta_stop or err == 0:
            if err == 0:
                print i
                print err
                break
            i += 1
            weak_learners.append(learner)
        self.learners = weak_learners
        self.learner_weights = learner_weights/np.sum(np.array(learner_weights))


    def sample(self, X, y, weights):
        X = np.array(X)
        y = np.array(y)
        results = np.random.choice(len(X),  3*len(X), p = weights) 
        sampledX = X[results]
        sampledY = y[results]
        return sampledX, sampledY

