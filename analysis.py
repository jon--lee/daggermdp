from svm import LinearSVM
from net import Net
from policy import SVMPolicy,NetPolicy
import numpy as np
import matplotlib.pyplot as plt
import IPython

from state import State
class Analysis():
    
    def __init__(self,H,W,ITERS):
        self.h = H
        self.w = W
        self.iters = ITERS
        self.density = np.zeros([H,W])


    def compute_std_er_m(self,data):
        n = data.shape[0]
        std = np.std(data)

        return std/np.sqrt(n)

    def compute_m(self,data):
        n = data.shape[0]
        return np.sum(data)/n

    def get_perf(self,data):
        #SAve each mean and err at the end
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


    def display_test_train(self,train,test):
        #Write a function to output test train.
        print "TEST LOSS ", np.sum(test_loss)/TRIALS
        print "TRAIN LOSS", np.sum(train_loss)/TRIALS
        #SAve 

    def plot(self):
        plt.ylabel('Reward')
        plt.xlabel('Iterations')

        names = ['NN_Supervise','LOG_Supervisor']
        plt.legend(names,loc='upper right')

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22},

        axes = plt.gca()
        axes.set_xlim([0,10])

        plt.show()

    def count_states(self,all_states):
        N = all_states.shape[0]
        current_density = np.zeros([self.h,self.w])
        for i in range(N):
            x = all_states[i,0]
            y = all_states[i,1]
            current_density[x,y] = current_density[x,y] +1.0

        norm = np.sum(current_density)
        IPython.embed()
        current_density = current_density/norm
        self.density = self.density+ current_density/self.iters

        

    def compile_density(self):
        density_r = np.zeros([self.h*self.w,3])
        norm = np.sum(self.density)
        self.m_val = 0.0
        for w in range(self.w):
            for h in range(self.h):
                val = self.density[w,h]
                if(val > self.m_val):
                    self.m_val = val
                val_r = np.array([h,w,val])
                density_r[w*self.w+h,:] = val_r
        print "M VAL ", self.m_val
        return density_r

    def show_states(self):
        plt.xlabel('X')
        plt.ylabel('Y')
        cm = plt.cm.get_cmap('gray_r')

        axes = plt.gca()
        axes.set_xlim([0,15])
        axes.set_ylim([0,15])
        density_r = self.compile_density()
        IPython.embed()
        plt.scatter(density_r[:,1],density_r[:,0], c= density_r[:,2]/self.m_val,cmap = cm,s=300,edgecolors='none')
        #save each density if called 
       
        #PLOT GOAL STATE
        plt.scatter([7],[7], c= 'green',s=300)

        #PLOT SINK STATE
        plt.scatter([4],[2], c= 'red',s=300)

        plt.show()


