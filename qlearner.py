from state import State
import numpy as np
from policy import QPolicy
from Net.tensor import qnet
from Net.tensor import inputdata
import tensorflow as tf
from sklearn.svm import SVR
class QLearner():
    
    def __init__(self, grid, mdp, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.moves = moves
        self.super_pi = mdp.pi
        self.reward = np.zeros(self.moves)
        self.animate = False 
        self.recent_rollout_states = None
        self.Q = MCfunc()
        self.alpha = .5
        self.gamma = .9
        self.states = []
    
    
    def guide2(self):
        self.grid.reset_mdp()
        self.reward = np.zeros(self.moves)
        self.recent_rollout_states = [self.mdp.state]
        recent_actions = []
        visits = []
        for t in range(self.moves):
            s = self.mdp.state
            a = int(self.super_pi.get_next(s))

            self.grid.step(self.mdp)
            s_prime = self.mdp.state
            self.reward[t] = self.grid.reward(s, a, s_prime)
            if not (str(s), a) in visits:
                self.recent_rollout_states.append(self.mdp.state)
                recent_actions.append(a)
            visits.append((str(s), a))
            #self.update(s, a, s_prime, self.reward[t])
        
        if self.animate:
            self.grid.show_recording()
        
        
        for t, s, a in zip(range(0, self.moves), self.recent_rollout_states, recent_actions):
                if t == 0:
                    self.Q.set(s, a, sum(self.reward[t:]))
                else:
                    self.Q.set(s, a, sum(self.reward[t-1:]))
        
        self.states += self.recent_rollout_states
    
    def guide(self):
        self.grid.reset_mdp()
        self.reward = np.zeros(self.moves)
        traj = []
        self.recent_rollout_states = [self.mdp.state]
        for t in range(self.moves):
            s = self.mdp.state
            a = int(self.super_pi.get_next(s))

            self.grid.step(self.mdp)
            s_prime = self.mdp.state
            self.reward[t] = self.grid.reward(s, a, s_prime)
            
            traj.append((s.x, s.y, a))
            self.recent_rollout_states.append(self.mdp.state)

        if self.animate:
            self.grid.show_recording()
        
        self.update_traj(traj, self.reward)
        self.states += self.recent_rollout_states
    

    def update_traj(self, traj, rewards):
        utraj = []
        for t, op in enumerate(traj):
            #if True:
            if not op in utraj:    
                x, y, a = op
                if t == 0:
                    self.Q.set(State(x, y), a, sum(rewards[t:]))
                else:
                    self.Q.set(State(x, y), a, sum(rewards[t-1:]))

                utraj.append(op)
                
    def rollout(self):
        self.grid.reset_mdp()
        self.reward = np.zeros(self.moves)
        traj = []
        self.recent_rollout_states = [self.mdp.state]
        for t in range(self.moves):
            s = self.mdp.state
            a = int(self.mdp.pi.get_next(s))

            

            self.grid.step(self.mdp)
            s_prime = self.mdp.state
            self.reward[t] = self.grid.reward(s, a, s_prime)

            traj.append((s.x, s.y, a))
            self.recent_rollout_states.append(self.mdp.state)

        if self.animate:
            self.grid.show_recording()
            
        self.update_traj(traj, self.reward)
        self.states += self.recent_rollout_states
        return


    def update(self, s, a, s_prime, r):
        q = self.Q.get(s, a)
        maxQ = self.gamma * max([self.Q.get(s_prime, a_prime) for a_prime in self.super_pi.available_actions])
        q_prime = q + self.alpha * (r + maxQ - q)
        self.Q.set(s, a, q_prime)

    def get_reward(self):
        return np.sum(self.reward)

    def retrain(self):
        self.Q.train()
        return

    def clear_states(self):
        self.states = []

    def return_states(self, state_list):
        N = len(state_list)
        states = np.zeros([N,2])
        for i in range(N):
            x = state_list[i].toArray()
            states[i,:] = x        
        return states

    def get_states(self):
        return self.return_states(self.states)

    def get_recent_rollout_states(self):
        return self.return_states(self.recent_rollout_states)

class Qfunc():


    def __init__(self):
        self.dataset = {}
        return
    
    def get(self, s, a):
        if not (str(s), a) in self.dataset:
            return -5.0
        return self.dataset[(str(s), a)]

    def set(self, s, a, val):
        self.dataset[(str(s), a)] = val
        return
    
class MCfunc():

    def __init__(self):
        self.dataset = {}
        return

    def get(self, s, a):
        if not (str(s), a) in self.dataset:
            return -5.0
        return np.mean(self.dataset[(str(s), a)])

    def set(self, s, a, val):
        if not (str(s), a) in self.dataset:
            self.dataset[(str(s), a)] = [-5.0]
        self.dataset[(str(s), a)].append(val)

class Qapprox():

    beta = 50.0    

    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.dataset = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.learner = qnet.QNet()
        return
    
    def preprocess(self, x, y, a):
        x = float(x) #/ float(self.w) - .5
        y = float(y) #/ float(self.h) - .5
        a = float(a) #- 1.0) / float(len(QPolicy.available_actions))
        return x, y, a


    # fix this (maybe monte carlo at the end of trajectory)
    def get(self, s, a):
        tup = self.preprocess(s.x, s.y, a)
        return self.predictor.predict([[s.x, s.y, a]])
        #print tup
        # with self.graph.as_default():
        #     return self.learner.dist([tup])[0]
        return 0
            
        

    def train(self):
        data = inputdata.QData(self.dataset)
        batch = data.next_train_batch(123)
        X = np.array(batch[0])
        y = np.array(batch[1])
        print X
        print y
        self.svr = SVR(kernel='rbf', C=1e3, gamma=.1)
        print "training"
        self.predictor = self.svr.fit(X, y.ravel())
        #self.predictor = self.svr.fit([[1, 2, 3], [3,2, 1],[1,3,4]], [1,2, 3])
        print "done"
        #for key in self.dataset.keys():
        #    print str(key) + " --> " + str(np.mean(self.dataset[key]))
        #with self.graph.as_default():
        #    self.learner = qnet.QNet()
        #    self.learner.optimize(5000, data)
        
        

    def set(self, s, a, val):
        tup = self.preprocess(s.x, s.y, a)
        val = val #/ float(Qapprox.beta)
        if not tup in self.dataset:
            self.dataset[tup] = [-5.0]
        self.dataset[tup].append(val)
        return
    
    
