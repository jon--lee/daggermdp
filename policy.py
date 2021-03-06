import numpy as np


class Action():
    NONE = -1
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    LEFT = {NORTH: WEST, SOUTH: EAST, WEST: SOUTH, EAST: NORTH}
    RIGHT = {NORTH: EAST, SOUTH: WEST, WEST: NORTH, EAST: SOUTH}

    @staticmethod
    def isVertical(direction):
        return direction == Action.NORTH or direction == Action.SOUTH

    @staticmethod
    def isHorizontal(direction):
        return direction == Action.EAST or direction == Action.WEST

    @staticmethod
    def arePerpendicular(d1, d2):
        return ((Action.isVertical(d1) and Action.isHorizontal(d2))
                or (Action.isHorizontal(d1) and Action.isVertical(d2)))

    @staticmethod
    def areParallel(d1, d2):
        return not self.arePerpendicular(d1, d2)


class DumbPolicy():

    def __init__(self, grid = None):
        return

    def get_next(self, state):
        if state.y == 14 and state.x == 14:
            return Action.NONE
        if state.y > state.x:
            if state.y == 14:
                return Action.EAST
            else:
                return Action.NORTH
        else:
            if state.x == 14:
                return Action.NORTH
            else:
                return Action.EAST
    



class ClassicPolicy():
    
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    desc = 'CLASSIC_POLICY'
    def __init__(self, grid):
        self.arr = np.zeros((grid.width, grid.height))
        return
        

    def get_next(self, state):
        #print self.arr.shape
        return self.arr[state.x, state.y]

    def update(self, state, action):
        self.arr[state.x, state.y] = action

class SVMPolicy():
    
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    
    def __init__(self, svm):
        self.svm = svm

    def get_next(self, state):
        return self.svm.predict([[state.x, state.y]])


class NetPolicy():
    
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    
    def __init__(self, net):
        self.net = net

    def get_next(self, state):
        return self.net.predict([[state.x, state.y]])


class BoostPolicy():
    
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    
    def __init__(self, boost):
        self.boost = boost

    def get_next(self, state):
        return self.boost.predict([[state.x, state.y]])


class SKPolicy():
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    
    def __init__(self, est):
        self.est = est

    def get_next(self, state):
        return self.est.predict([[state.x, state.y]])
   



class QPolicy():
    
    available_actions = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST, Action.NONE}
    def __init__(self, qlearner):
        self.q = qlearner

    def get_next(self, s):
        max_a = None
        for a in QPolicy.available_actions:
            #print self.q.Q.get(s, a)
            if max_a is None or self.q.Q.get(s, a) > self.q.Q.get(s, max_a):
                max_a = a
        #print "Greedy: " + str(self.q.Q.get(s, max_a))
        return max_a
        #return self.q.predict([[state.x, state.y]])

