from policy import Action, DumbPolicy
from state import State
from mdp import ClassicMDP
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
class BasicGrid():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mdp = None
        self.time_steps = 0
        self.record_states = []
        return

    def add_mpd(self, mdp):
        if self.mdp is not None:
            self.mdp.grid = None
        self.mdp = mdp
        self.mdp.grid = self
        self.record_states = [self.mdp.state]

    def get_dir(self, state, state_prime):
        if state.x == state_prime.x and state.y + 1 == state_prime.y:
            return Action.NORTH
        elif state.x == state_prime.x and state.y - 1 == state_prime.y:
            return Action.SOUTH
        elif state.y == state_prime.y and state.x + 1 == state_prime.x:
            return Action.EAST
        elif state.y == state_prime.y and state.x - 1 == state_prime.x:
            return Action.WEST
        else:
            return Action.NONE

    def is_valid(self, state):
        return (state.y < self.height and state.y >= 0
                and state.x < self.width and state.x >= 0)

    def get_state_prime(self, state, action):
        state_prime = State(state.x, state.y)
        if action == Action.NORTH:
            state_prime.y = state.y + 1
        elif action == Action.SOUTH:
            state_prime.y = state.y - 1
        elif action == Action.EAST:
            state_prime.x = state.x + 1
        elif action == Action.WEST:
            state_prime.x = state.x - 1
        else:
            return state
        return state_prime

    def get_adjacent(self, state):
        north_state = self.get_state_prime(state, Action.NORTH)
        south_state = self.get_state_prime(state, Action.SOUTH)
        east_state = self.get_state_prime(state, Action.EAST)
        west_state = self.get_state_prime(state, Action.WEST)
        return (north_state, south_state, east_state, west_state)

    
    def reward(self, state, action, state_prime):
        if not self.is_valid(state_prime):
            state_prime = state
        if (state_prime.x == self.width - 1 and
                state_prime.y == self.height - 1):
            return 10
        elif (state_prime.x == self.width / 2 and
                state_prime.y == self.height / 2):
            return -10
        else:
            return -.02

    def update_world(self):
        #broken do not use
        #plot = plt.figure()
        
        #plt.plot([self.mdp.state.x], [self.mdp.state.y], 'ro')
        #plt.show(block=False)
        #plt.close()
        return


    def step(self):
        self.record_states.append(mdp.state)
        mdp.move()
        self.time_steps += 1
        
    def animate(self, i):
        #if i > 20:
        #    raise Exception
        #pullData = open("sampleText.txt","r").read()
        #dataArray = pullData.split('\n')
        self.figure.autoscale(False)
        if i < len(self.record_states):
            xar = [self.record_states[i].x]
            yar = [self.record_states[i].y]
            
            self.figure.clear()
            self.figure.scatter(xar,yar,s=15000 / self.height / self.width)
            self.figure.set_xlim([-.5, self.width - 1 +.5])

            width_range = np.arange(self.width)
            height_range = np.arange(self.height)

            plt.xticks(width_range)
            plt.yticks(height_range)
            
            plt.title("Step " + str(i))
            self.figure.set_yticks(height_range[:-1] + 0.5, minor=True)
            self.figure.set_xticks(width_range[:-1] + 0.5, minor=True)            
            self.figure.grid(which='minor', axis='both', linestyle='-')            
            self.figure.set_ylim([-.5, self.height - 1 + .5])                    

        

    def show_recording(self):
        fig, self.figure = plt.subplots()
        interval = 200
        try:
            an = animation.FuncAnimation(fig, self.animate, interval=interval, repeat=False)
            plt.show(block=False)
            plt.pause(interval * (len(self.record_states) + 1) / 1000)
        except:
            return
if __name__ == '__main__':
    state = State(0, 0)
    grid = BasicGrid(10, 10)
    
    mdp = ClassicMDP(DumbPolicy(), grid)
    moves = 1000
    for _ in range(moves):
        grid.step()
    grid.show_recording()
    
    
