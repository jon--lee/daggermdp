import numpy as np
from policy import Action, ClassicPolicy
from state import State
import state
import random
import pickle
import IPython


class ClassicMDP():

    gamma = .99

    def __init__(self, policy, grid):
        self.state = State(0, 0)
        self.grid = grid
        self.grid.add_mdp(self)
        self.pi = policy
        self.values = np.zeros((self.grid.width, self.grid.height))
        self.pi_noise = False
        self.noise = 0.5

        

    def transition_prob(self, state, action, state_prime):
        """
            Given a state obj, action (direction), and state',
            return the probability [0, 1] of a successful action
        """
        prime_dir = self.grid.get_dir(state, state_prime)
        if action == Action.NONE and prime_dir == action:
            return 1.0
        elif action == Action.NONE:
            return 0.0
        
        if prime_dir == Action.NONE:
            return 0.0
        elif action == prime_dir:
            return 0.8
        elif Action.arePerpendicular(action, prime_dir):
            return 0.1
        else:
            return 0.0
            
    def update_state(self, new_state):
        if self.grid.is_valid(new_state):
            self.state = new_state

    def rand_action(self): 
        actions = [-1.0,0.0,1.0,2.0,3.0]
        idx = np.random.randint(5)
        return actions[idx]

    def move(self):
        next_action = self.pi.get_next(self.state)    
        adjs = self.grid.get_adjacent(self.state)

        if(self.pi_noise):
            x = random.random()
            if(x < self.noise):
                print "NOISY"
                next_action = self.rand_action()


        x = random.random()
        prob_sum = 0.0
        for adj in adjs:
            trans_prob = self.transition_prob(self.state, next_action, adj)
            prob_sum += trans_prob
            if x < prob_sum:
                self.update_state(adj)
                return [adj, next_action]
        
        return [self.state,next_action]    
        

    def value_iteration(self):
        print "Performing value iteration"
        new_values = np.zeros((self.grid.width, self.grid.height))
        i = 0
        # emulating a do-while loop (see break condition at bottom)
        while True:
            delta = 0.0            
            for state in self.grid.get_all_states():
                curr_val = self.value(state)
                new_val = 0.0
                for action in self.pi.available_actions:
                    action_sum = 0.0
                    for adj in self.grid.get_adjacent(state):
                        action_sum += (self.transition_prob(state, action, adj)
                                * (self.grid.reward(state, action, adj) + 
                                    self.gamma * self.value(adj)))
                    new_val = max(new_val, action_sum)
                self.set_value(state, new_val, self.values)
                delta = max(delta, abs(curr_val - new_val))
            i += 1
            if delta < 1e-2:
                print "Iterations: " + str(i)
                break

        # find actions with current value
        for state in self.grid.get_all_states():
            best_action = Action.NONE
            best_value = 0.0
            for action in self.pi.available_actions:
                action_sum = 0.0
                for adj in self.grid.get_adjacent(state):
                    action_sum += (self.transition_prob(state, action, adj)
                                * (self.grid.reward(state, action, adj) + 
                                    self.gamma * self.value(adj)))
                if action_sum > best_value:
                    best_value = action_sum
                    best_action = action
            self.pi.update(state, best_action)
        return

    def value(self, state):
        #TODO: change this to reflect nearest valid state (not current state)
        if not self.grid.is_valid(state):
            state = self.grid.get_nearest_valid(state)
        return self.values[state.x, state.y]
    
    def set_value(self, state, value, values):
        #TODO: see above
        if not self.grid.is_valid(state):
            state = self.grid.get_nearest_valid(state)
        values[state.x, state.y] = value

    

    def save_policy(self, filename='policy.p'):
        f = open(filename, 'w')
        pickle.dump(self.pi, f)
        f.close()

    def load_policy(self, filename='policy.p'):
        f = open(filename, 'r')
        self.pi = pickle.load(f)
        f.close()
