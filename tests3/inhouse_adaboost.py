from adaboost import Adaboost
from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP
from svm_dagger import SVMDagger
from boost_supervise import BoostSupervise 
from gridworld import BasicGrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from dagger import Dagger
from supervise import Supervise
from nsupervise import NSupervise
from analysis import Analysis
import IPython
import plot_class
import scenarios
from svm_supervise import SVMSupervise
from scikit_supervise import ScikitSupervise
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifierCV, Perceptron, SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, NuSVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import itertools
import os

short = 'comparisons2/inhouse_adaboost_'
comparisons_directory = short + 'comparisons/'
data_directory = short + 'data/'

comparisons_directory, data_directory = make_name(ne, lr)
if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

def fit_class_SVC(data, delta_stop):
	'''
    performs the adaboost code using classification
    '''
    features = np.array([point[0] for point in data])
    labels = np.array([labels[1] for point in data])
    weak_learners = []
    learner_weights = []
    data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
    i = 0
    while i < 100:
        learner = SVC() # instantiate a new weak learner
        learner.optimize(X, labels, data_weights)
        err = learner.predict(X, labels, data_weights) 
        # results = learner.accuracy(data_embedding) # results is expected to an array of classes
        raw_errors = np.equal(results, labels)
        err = np.sum((1-raw_errors)  * data_weights)

        # get the weight of the learner
        learner_weight = np.log2(phi*(1-err)/(err*(1-phi)))
        learner_weights.append(learner_weight)

        data_weights = data_weights*np.exp(-learner_weight*raw_errors)
        normalization = np.sum(data_weights)
        data_weights = data_weights/normalization

        if err >= phi - delta_stop or err == 0:
            break
        i += 1
        weak_learners.append(learner)

    ensemble = weak_learners
    weights = learner_weights/np.sum(np.array(learner_weights))