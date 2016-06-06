from svm import LinearSVM
#from net import Net
from mpl_toolkits.mplot3d import Axes3D
#from policy import SVMPolicy,NetPolicy
import numpy as np
import matplotlib.pyplot as plt
import IPython
import cPickle

from state import State
class Plotter():
    
    def __init__(self,H=15,W=15,ITERS=1,rewards=None, sinks=None, desc="No description"):
        self.h = H
        self.w = W
        self.iters = ITERS
        self.density = np.zeros([H,W])

        self.desc = desc
        self.test_loss = -1.0
        self.train_loss = -1.0
        self.x = None
        self.mean = None
        self.err = None


    def count_data(self,data):
        N = data.shape[0]
        current_density = np.zeros([self.h,self.w])
        for i in range(N):
            x = data[i][0][0]
            y = data[i][0][1]
          
            current_density[x,y] = current_density[x,y] +1.0
        
        norm = np.sum(current_density)

        current_density = current_density/norm
        self.density = current_density


    def count_states(self,all_states):
        N = all_states.shape[0]
        current_density = np.zeros([self.h,self.w])
        for i in range(N):
            x = all_states[i,0]
            y = all_states[i,1]
            current_density[x,y] = current_density[x,y] +1.0
        
        norm = np.sum(current_density)

        current_density = current_density/norm
        self.density = current_density

        

    def compile_density(self):
        density_r = np.zeros([self.h*self.w,3])

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


    def plot_scatter(self,weights=None,color='density'):
        plt.xlabel('X')
        plt.ylabel('Y')
        cm = plt.cm.get_cmap('gray_r')

        axes = plt.gca()
        axes.set_xlim([0,15])
        axes.set_ylim([0,15])
        density_r = self.compile_density()

        #print density_r
        #print np.sum(density_r)
        if(color == 'density'):
            a = np.copy(density_r[:,2])
            
            plt.scatter(density_r[:,1],density_r[:,0], c= a, cmap = cm,s=300,edgecolors='none') 
        else: 
            a = weights
            plt.scatter(weights[:,0],weights[:,1], c= weights[:,2], cmap = cm,s=300) 
            
        #plt.scatter(density_r[:,1],density_r[:,0], c= density_r[:,2],cmap = cm,s=300,edgecolors='none', color='blue')

        #save each density if called 
       
        
        # #PLOT GOAL STATE
        # plt.scatter([7],[7], c= 'green',s=300)

        # #PLOT SINK STATE
        # plt.scatter([4],[2], c= 'red',s=300)

        plt.show()

    def show_states(self):
        self.plot_scatter()


    def show_weights(self,weights):
        self.plot_scatter(weights=weights,color = 'weights')

