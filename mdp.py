import numpy as np
from policy import Action
from state import State
import state
import random

        


class ClassicMDP():


    def __init__(self, policy, grid):
        self.state = State(0, 0)
        self.grid = grid
        self.grid.add_mpd(self)
        self.pi = policy
        

    def transition_prob(self, state, action, state_prime):
        """
            Given a state obj, action (direction), and state',
            return the probability [0, 1] of a successful action
        """
        prime_dir = self.grid.get_dir(state, state_prime)
        if prime_dir == Action.NONE:
            return 0.0
        elif action == prime_dir:
            return 0.8
        elif Action.arePerpendicular(action, prime_dir):
            return .1
        else:
            return 0.0
            
    def update_state(self, new_state):
        if self.grid.is_valid(new_state):
            self.state = new_state


    def move(self):
        next_action = self.pi.get_next(self.state)
        adjs = self.grid.get_adjacent(self.state)
        x = random.random()
        prob_sum = 0.0
        for adj in adjs:
            trans_prob = self.transition_prob(self.state, next_action, adj)
            prob_sum += trans_prob
            if x < prob_sum:
                self.update_state(adj)
                return adj
        return self.state    
        

    def value_iteration(self):
         

