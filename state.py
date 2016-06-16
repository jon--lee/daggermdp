import numpy as np

class State():
    def __init__(self, x, y):
        self.x = x;
        self.y = y;

    def toString(self):
        return "(State x: " + str(self.x) + ", y: " + str(self.y) + ")"

    def toArray(self): 
    	return np.array([self.x,self.y])

    def __repr__(self):
        return self.toString()
            
    def __str__(self):
        return self.toString()

    def equals(self, other):
        return self.x == other.x and self.y == other.y

    def __eq__(self, other):
        return self.equals(other)
