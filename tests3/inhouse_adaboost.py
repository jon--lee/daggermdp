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

plotter = plot_class.Plotter()


short = 'comparisons3/inhouse_adaboost_'
comparisons_directory = short + 'comparisons/'
data_directory = short + 'data/'

if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)



ITER = 25
TRIALS = 15
SAMP = 20
LIMIT_DATA = 1
DEPTH = 6

H = 15
W = 15

grid = BasicGrid(H, W)
rewards = scenarios.scenario3['rewards']
sinks = scenarios.scenario3['sinks']
grid.reward_states = rewards
grid.sink_states = sinks

mdp = ClassicMDP(ClassicPolicy(grid), grid)
#mdp.value_iteration()
#mdp.save_policy('scen4.p')
mdp.load_policy('scen4.p')

value_iter_pi = mdp.pi
plotter.plot_state_actions(value_iter_pi, rewards = grid.reward_states, sinks = grid.sink_states,
        filename=comparisons_directory + 'value_iter_state_action.png')

value_iter_data = np.zeros([TRIALS, ITER])
svm_il_data = np.zeros([TRIALS, ITER])
svm_il_acc = np.zeros([TRIALS, ITER])
svm_il_loss = np.zeros([TRIALS, ITER])



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
        while i < 10:
            learner = SVC(kernel='linear') # instantiate a new weak learner
            #learner = DecisionTreeClassifier(max_depth=3)
            learner.fit(X, labels, sample_weight=data_weights)
            err = learner.predict(X) 
            # results = learner.accuracy(data_embedding) # results is expected to an array of classes
            raw_errors = np.equal(err, labels)
            err = np.sum((1-raw_errors)  * data_weights)

            # get the weight of the learner
            learner_weight = np.log2(self.phi*(1-err)/(err*(1-self.phi)))
            learner_weights.append(learner_weight)

            data_weights = data_weights*np.exp(-learner_weight*raw_errors)
            normalization = np.sum(data_weights)
            data_weights = data_weights/normalization

            if err >= self.phi - self.delta_stop or err == 0:
                break
            i += 1
            weak_learners.append(learner)

        self.learners = weak_learners
        self.learner_weights = learner_weights/np.sum(np.array(learner_weights))

    def predict(self, X):
        predictions = np.zeros(len(X))
        for learner, weight in zip(self.learners, self.learner_weights):
            predictions += learner.predict(X) * weight
        prediction_values = np.around(predictions)
        prediction_values[prediction_values > 3] = 3
        prediction_values[prediction_values < -1] = -1
        return prediction_values





for t in range(TRIALS):
    print "\nIL Trial: " + str(t)
    mdp.load_policy('scen4.p')

    svm = SVC(kernel='linear')
    #dt = DecisionTreeClassifier(max_depth=3)
    sup = ScikitSupervise(grid, mdp, Classifier=svm)
    sup.sample_policy()

    value_iter_analysis = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Value iter policy')
    value_iter_r = np.zeros(ITER)
    svm_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)
    loss = np.zeros(ITER)

    sup.record = True

    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi 
        sup.record = True
        for _ in range(SAMP):
            if _  >= LIMIT_DATA:
                sup.record = False
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / (SAMP)

        sup.record = False
        print "     Training on " + str(len(sup.net.data)) + " examples"
        sup.train()

        acc[i] = sup.svm.acc()
        for _ in range(SAMP):
            sup.record=False
            sup.rollout()
            loss[i] += sup.get_loss() / float(SAMP)
            svm_il_r[i] += sup.get_reward() / SAMP
        #print acc        
    if t == 0:
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename=comparisons_directory + 'svm_il_state_action.png')        
    
    
    svm_il_data[t,:] = svm_il_r
    value_iter_data[t,:] = value_iter_r
    svm_il_acc[t,:] = acc
    svm_il_loss[t,:] = loss




#BOOSTED SUPERVISOR

ada_il_data = np.zeros([TRIALS, ITER])
ada_il_acc = np.zeros([TRIALS, ITER])
ada_il_loss = np.zeros([TRIALS, ITER])


for t in range(TRIALS):
    print "\nAdaboost IL Trial: " + str(t)
    mdp.load_policy('scen4.p')

    #svm = SVC(kernel='linear')
    #ada = AdaBoostClassifier(svm, n_estimators=5, algorithm='SAMME')
    ada = SVC_boosted()
    sup = ScikitSupervise(grid, mdp, Classifier=ada)
    sup.sample_policy()

    value_iter_analysis = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Value iter policy')
    ada_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)
    loss = np.zeros(ITER)

    sup.record = True
    #for _ in range(4):
    #    sup.rollout()

    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi 
        sup.record = True
        for _ in range(SAMP):
            if _  >= LIMIT_DATA:
                sup.record = False
            sup.rollout()

        sup.record = False
        print "     Training on " + str(len(sup.net.data)) + " examples"
        sup.train()

        acc[i] = sup.svm.acc()
        for _ in range(SAMP):
            sup.record=False
            sup.rollout()
            loss[i] += sup.get_loss() / float(SAMP)
            ada_il_r[i] += sup.get_reward() / SAMP
        #print acc        
    if t == 0:
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename=comparisons_directory + 'ada_il_state_action.png')        
    ada_il_data[t,:] = ada_il_r
    ada_il_acc[t,:] = acc
    ada_il_loss[t,:] = loss


# SKLEARN adaboost

sk_il_data = np.zeros([TRIALS, ITER])
sk_il_acc = np.zeros([TRIALS, ITER])
sk_il_loss = np.zeros([TRIALS, ITER])


for t in range(TRIALS):
    print "\nSKLearn adaboost IL Trial: " + str(t)
    mdp.load_policy('scen4.p')

    svm = SVC(kernel='linear')#DecisionTreeClassifier(max_depth=3)#SVC(kernel='linear')
    ada = AdaBoostClassifier(svm, n_estimators=10, algorithm='SAMME')
    #ada = SVC_boosted()
    sup = ScikitSupervise(grid, mdp, Classifier=ada)
    sup.sample_policy()

    value_iter_analysis = Analysis(W, H, ITER, rewards=rewards, sinks=sinks,
            desc='Value iter policy')
    sk_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)
    loss = np.zeros(ITER)

    sup.record = True
    #for _ in range(4):
    #    sup.rollout()

    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi 
        sup.record = True
        for _ in range(SAMP):
            if _  >= LIMIT_DATA:
                sup.record = False
            sup.rollout()

        sup.record = False
        print "     Training on " + str(len(sup.net.data)) + " examples"
        sup.train()

        acc[i] = sup.svm.acc()
        for _ in range(SAMP):
            sup.record=False
            sup.rollout()
            loss[i] += sup.get_loss() / float(SAMP)
            sk_il_r[i] += sup.get_reward() / SAMP
        #print acc        
    if t == 0:
        plotter.plot_state_actions(mdp.pi, rewards=rewards, sinks=sinks,
                filename=comparisons_directory + 'sk_il_state_action.png')        
    sk_il_data[t,:] = sk_il_r
    sk_il_acc[t,:] = acc
    sk_il_loss[t,:] = loss







# print value_iter_data
# print classic_il_data
# print dagger_data

np.save(data_directory + 'sup_data.npy', value_iter_data)
np.save(data_directory + 'svm_classic_il_data.npy', svm_il_data)
np.save(data_directory + 'ada_classic_il_data.npy', ada_il_data)
np.save(data_directory + 'sk_classic_il_data.npy', ada_il_data)


np.save(data_directory + 'ada_il_acc.npy', ada_il_acc)
np.save(data_directory + 'svm_il_acc.npy', svm_il_acc)
np.save(data_directory + 'sk_il_acc.npy', svm_il_acc)

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(svm_il_data)
analysis.get_perf(ada_il_data)
analysis.get_perf(sk_il_data)

#analysis.plot(names = ['Value iteration', 'Adaboost IL'], filename=comparisons_directory + 'svm_reward_comparison.png', ylims=[-60, 100])
analysis.plot(names = ['Value iteration', 'LSVM IL', 'LSVM Boosted IL', 'LSVM SK Boosted'], filename=comparisons_directory + 'svm_reward_comparison.png', ylims=[-60, 100])
print "Saving analysis to: " + comparisons_directory + 'svm_reward_comparison.png'

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
acc_analysis.get_perf(svm_il_acc)
acc_analysis.get_perf(ada_il_acc)
acc_analysis.get_perf(sk_il_acc)

acc_analysis.plot(names = ['LSVM Acc.', 'LSVM Boosted Acc.', 'LSVM SK Boosted'], label='Accuracy', filename=comparisons_directory + 'svm_acc_comparison.png', ylims=[0,1])
#acc_analysis.plot(names = ['Adaboost IL Acc.'], label='Accuracy', filename=comparisons_directory + 'svm_acc_comparison.png', ylims=[0,1])

loss_analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
loss_analysis.get_perf(svm_il_loss)
loss_analysis.get_perf(ada_il_loss)
loss_analysis.get_perf(sk_il_loss)
loss_analysis.plot(names = ['LSVM loss', 'LSVM Boosted loss', 'LSVM SK Boosted loss'], label='Loss', filename=comparisons_directory + 'loss_plot.png', ylims=[0, 1])






