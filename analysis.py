from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np
import matplotlib.pyplot as plt
import IPython

from state import State
class Analysis():
        
    def compute_std_er_m(self,data):
        n = data.shape[0]
        std = np.std(data)

        return std/np.sqrt(n)

    def compute_m(self,data):
        n = data.shape[0]
        return np.sum(data)/n

    def get_perf(self,data):
        iters = data.shape[1]
        mean = np.zeros(iters)
        err = np.zeros(iters)
        x = np.zeros(iters)

        for i in range(iters):
            mean[i] = self.compute_m(data[:,i])
            x[i] = i
            err[i] = self.compute_std_er_m(data[:,i])
        
        plt.errorbar(x,mean,yerr=err,linewidth=5.0)
    
        return [mean,err]

    def plot(self):
        plt.ylabel('Reward')
        plt.xlabel('Iterations')

        names = ['DAgger','Supervise','N_Supervise']
        plt.legend(names,loc='upper right')

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}

        axes = plt.gca()
        axes.set_xlim([0,10])

        plt.show()



