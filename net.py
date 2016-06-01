from sklearn import svm
import tensorflow as tf
import numpy as np
import IPython
        
import sys
sys.path.append("../")

from Net.tensor import gridnet
from Net.tensor import gridlog
from Net.tensor import inputdata

class Net():

    def __init__(self, grid, mdp, net_name = 'Net'):
        self.mdp = mdp
        self.grid = grid
        self.net_name = net_name
        self.data = []
        self.svm = None

    def add_datum(self, state, action):
        self.data.append((state, action))

    def clear_data(self):
        self.data = []
    
    def get_states(self):
        N = len(self.data)
        states = np.zeros([N,2])
        for i in range(N):
            x = self.data[i][0].toArray()
            states[i,:] = x

        return states


    def fit(self):
        data = inputdata.GridData(self.data)
        
        g = tf.Graph()          # this appears to improve speed when
        with g.as_default():    # initializing many times by avoiding re-initializing variables
            if(self.net_name == 'Net'):
                self.net = gridnet.GridNet()
            elif(self.net_name == 'Log'):
                self.net = gridlog.GridLog()
            #self.net.optimize(2000,data,batch_size = 50)
            self.net.optimize(1000,data,batch_size = 50)
    
    def predict(self, a):
        return self.net.predict(a)

    def return_stats(self):
        return self.net.get_stats()
