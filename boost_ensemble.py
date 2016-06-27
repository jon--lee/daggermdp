import tensorflow as tf
import numpy as np
from net import Net
from Net.tensor import gridnet

class boostEnsemble():

	def __init__(self, ensemble_name = 'Net', delta_stop, net):
        self.ensemble_name = ensemble_name
        self.delta_stop = delta_stop
        self.net = net

    def fit(self):


    def get_weights(self):
        data = inputdata.GridData_UB(self.data,self.T)
        return data.get_weights(self.net)
    
    def predict(self, a):
        return self.net.predict(a)

    def return_stats(self):
        return self.net.get_stats()