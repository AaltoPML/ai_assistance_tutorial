import numpy as np
import copy
from tqdm import tqdm
import pickle
# from copy import deepcopy as cpy


class StateNode:ç
    def __init__(self, state=None, parent=None, is_root=False, is_final=False):
        self.n_visits = 0
        self.reward = 0
        self.state = state
        self.parent = parent
        self.is_final = is_final
        self.is_root = is_root
        self.children = {}


    def add_children(self, action_node):
        self.children[action_node.action] = action_node


    def next_action_node(self, action):
        if action not in self.children.keys():
            new_action_node = ActionNode(action, parent=self)
            self.add_children(new_action_node)
        else:
            new_action_node = self.children[action]
        return new_action_node




class ActionNode:
    def __init__(self, action, parent=None):
        self.n_visits = 0
        self.cumulative_reward = 0
        self.action = action
        self.parent = parent
        self.children = {}


    def add_children(self, state_node):
        self.children[state_node.state] = state_node




class BAMCTS:

    def __init__(self, initial_obs, env, K, action_space=None):
        # Maybe it's better to initialize the node reward by the GP_AI
        self.env = env
        self.K = K
        self.root = StateNode(state=initial_obs, is_root=True)


    def plan(self, n_sim, progress_bar=False):
        if progress_bar:
            iterations = tqdm(range(n_sim))
        else:
            iterations = range(n_sim)

        for _ in iterations:
            self.grow_tree()
            
    
    def grow_tree(self):
        state_node = self.root
        self.env.reset() # resample user_model
        internal_env = copy.copy(self.env)

        while (not state_node.is_final) and state_node.n_visits > 1:

            a = self.select_action(state_node)
            new_action_node = state_node.next_action_node(a)

            new_state_node, r = self.get_outcome(internal_env, new_action_node)
            new_state_node = self.update_state_node(new_state_node, new_action_node)

            new_state_node.reward = r
            new_action_node.reward = r

            state_node = new_state_node

        state_node.n_visits += 1
        cumulative_reward = self.evaluate(internal_env)

        while not state_node.is_root:
            action_node = state_node.parent
            cumulative_reward += action_node.reward
            action_node.cumulative_reward += cumulative_reward
            action_node.n_visits += 1
            state_node = action_node.parent
            state_node.n_visits += 1
        
    
    def select_action(self, state_node):
        if state_node.n_visits <= 2:
            state_node.children = {a: ActionNode(a, parent=state_node) for a in self.env.action_space}

        def scoring(k):
            if state_node.children[k].n_visits > 0:
                return state_node.children[k].cumulative_reward/state_node.children[k].n_visits + \
                    self.K*np.sqrt(np.log(state_node.n_visits)/state_node.children[k].n_visits)
            else:
                return np.inf

        a = max(state_node.children, key=scoring)

        return a
    
    
    def get_outcome(self, env, action_node):
        new_state_index, r, done = env.step(action_node.action)        
        return StateNode(state=new_state_index, parent=action_node, is_final=done), r

    
    def update_state_node(self, state_node, action_node):
        if state_node.state not in action_node.children.keys():
            state_node.parent = action_node
            action_node.add_children(state_node)
        else:
            state_node = action_node.children[state_node.state]

        return state_node


    def evaluate(self, env): # this function should be refined. it cannot go until the end in our case
        """
        Evaluates a state node by playing to a terminal node using the rollot policy
        """
        max_iter = 10
        R = 0
        done = False
        iter = 0
        while ((not done) and (iter < max_iter)):
            iter += 1
            a = np.random.choice(env.action_space)
            s, r, done = env.step(a)
            R += r
        
        return R

    
    def find_best_action(self):
        """
        At the end of the simulations returns the most visited action
        """
        actions = [node.action for node in self.root.children.values() if node.n_visits]
        number_of_visits_children = [node.n_visits for node in self.root.children.values() if node.n_visits]
        value_children = [node.cumulative_reward for node in self.root.children.values() if node.n_visits]
        mean_value_children = [node.cumulative_reward/node.n_visits for node in self.root.children.values() if node.n_visits]
        index_best_action = np.argmax(mean_value_children)
        best_action = actions[index_best_action]
        """
        indx = np.argpartition(mean_value_children, -6)[-6:]

        print("Best:", index_best_action, mean_value_children[index_best_action])
        print("Best actions:", indx)
        print("Values:", [value_children[i] for i in indx])
        print("mean values:", [mean_value_children[i] for i in indx])
        print("Visits:", [number_of_visits_children[i] for i in indx])
        print("Actions:", [actions[i] for i in indx])
        """
        
        return best_action



