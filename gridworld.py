from policy import Action, DumbPolicy, ClassicPolicy
from state import State
from mdp import ClassicMDP
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from dagger import Dagger
class BasicGrid():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mdp = None
        self.time_steps = 0
        self.record_states = []
        self.reward_state = State(width  / 2, height / 2)
        self.failure_state = State(4, 2)
        return

    def add_mdp(self, mdp):
        if self.mdp is not None:
            self.mdp.grid = None
        self.mdp = mdp
        self.mdp.grid = self
        self.mdp.state = State(0, 0)
        self.record_states = [self.mdp.state]
        self.time_steps = 0

    def reset_mdp(self):
        self.add_mdp(self.mdp)

    def clear_record_states(self):
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
        same_state = self.get_state_prime(state, Action.NONE)
        return (north_state, south_state, east_state, west_state, same_state)

    def get_nearest_valid(self, state):
        if self.is_valid(state):
            print "is_valid"
            return state
        new_state = State(state.x, state.y)
        new_state.x = max(0, state.x)
        new_state.x = min(self.width - 1, new_state.x)
        new_state.y = max(0, new_state.y)
        new_state.y = min(self.height - 1, new_state.y)
        return new_state
    
    def reward(self, state, action, state_prime):
        #TODO: change invalid state_primes to reflect nearest valid state (not current state)
        if not self.is_valid(state_prime):
            state_prime = state
        if (state_prime.x == self.reward_state.x and
                state_prime.y == self.reward_state.y):
            return 10
        elif (state_prime.x == self.failure_state.x and
                state_prime.y == self.failure_state.y):
            return -10
        else:
            return -.02

    def _draw_reward(self, size):
        self.figure.scatter([self.reward_state.x], [self.reward_state.y], s=size, c='g') 
        return
    
    def _draw_failure(self, size):
        self.figure.scatter([self.failure_state.x], [self.failure_state.y], s=size, c='r') 
        return

    
    def step(self):
        self.record_states.append(mdp.state)
        mdp.move()
        self.time_steps += 1
        
    def _animate(self, i):
        self.figure.autoscale(False)
        if i < len(self.record_states):
            xar = [self.record_states[i].x]
            yar = [self.record_states[i].y]
            robo_size = 15000 / self.height / self.width
            indicator_size = 30000 / self.height / self.width
            self.figure.clear()
            self._draw_reward(indicator_size)
            self._draw_failure(indicator_size)
            self.figure.scatter(xar,yar, s=robo_size)            
            self.figure.set_xlim([-.5, self.width - 1 +.5])
            self.figure.set_ylim([-.5, self.height - 1 + .5])                                

            width_range = np.arange(self.width)
            height_range = np.arange(self.height)

            plt.xticks(width_range)
            plt.yticks(height_range)
            
            plt.title("Step " + str(i))
            self.figure.set_yticks((height_range[:-1] + 0.5), minor=True)
            self.figure.set_xticks((width_range[:-1] + 0.5), minor=True)            
            self.figure.grid(which='minor', axis='both', linestyle='-')            

        

    def show_recording(self):
        fig, self.figure = plt.subplots()
        # All recordings should take ~10 seconds
        interval = float(5000) / float(len(self.record_states))
        try:
            an = animation.FuncAnimation(fig, self._animate, interval=interval, repeat=False)
            plt.show(block=False)
            plt.pause(interval * (len(self.record_states) + 1) / 1000)
            plt.close()
        except:
            return

    def get_all_states(self):
        for i in range(self.width):
            for j in range(self.height):
                yield State(i, j)

        
if __name__ == '__main__':
    grid = BasicGrid(15, 15)
    mdp = ClassicMDP(ClassicPolicy(grid), grid)
    #mdp.value_iteration()
    #mdp.save_policy()
    mdp.load_policy()
    #for i in range(40):
    #    grid.step()
    #grid.show_recording()
    
    dagger = Dagger(grid, mdp)
    dagger.rollout()            # rollout with supervisor policy
    for _ in range(5):
        dagger.retrain()
        dagger.rollout()
