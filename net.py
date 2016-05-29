from sklearn import svm
import tensorflow as tf
import IPython
        
import sys
sys.path.append("../")

from Net.tensor import gridnet
from Net.tensor import inputdata

class Net():

    def __init__(self, grid, mdp):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.svm = None

    def add_datum(self, state, action):
        self.data.append((state, action))
        
    def fit(self):
        data = inputdata.GridData(self.data)
      
        self.net = gridnet.GridNet()
        #self.net.optimize(2000,data,batch_size = 50)
        self.net.optimize(1000,data,batch_size = 50)
    
    def predict(self, a):
        return self.net.predict(a)
