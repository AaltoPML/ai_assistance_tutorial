import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import copy

# Utils
def dec2bin(n, n_digs=None):
    num = bin(n).replace("0b","")
    if n_digs:
        num = "0"*(n_digs-len(num))+num
    return num
    

def bin2dec(n):
    return int(n,2)

# Helper objects
class Route:
    def __init__(self, start, end, mode=None,
                    price=None, time=None, dist=None):
        self.start = start
        self.end = end
        self.mode = mode
        self.price = price
        self.time = time
        self.dist = dist
    
    def __eq__(self, other):
        if (isinstance(other, Route)):
            return vars(self) == vars(other)
        return False
    
    def __repr__(self):
         return str(vars(self))
    def __str__(self):
         return str(self.get_itinerary())
    
    def get_itinerary(self):
        return (self.start,self.end,self.mode)
    
    def get_costs(self):
        return (self.dist,self.price,self.time)
    

# Main AI assistnece objects
class World:
    def __init__(self, n_cities, n_modes, modes_prob, modes_price, modes_time, modes_dist):
        self.n_cities = n_cities
        self.n_modes = n_modes
        self.modes_prob = modes_prob
        self.bin_edges = None
        self.graph = nx.Graph()
        self._generate_random_map()
        self.start, self.destination = np.random.choice(self.n_cities, size=(2,), replace=False)
        self.path_ai, self.path_user = None, None
        # generate price, duration and other factors for the graph
        self.prices = self._generate_properties(modes_price)
        self.times = self._generate_properties(modes_time)
        self.dists = self._generate_properties(modes_dist)
        self._find_best_pos()


    def step(self, ai_advice=None, user_action=None):
        if ai_advice is not None and self._is_valid(ai_advice):
            self.path_ai = []
            for x,y,m in ai_advice:
                self.path_ai.append(Route(x, y, m, price=self.prices[x,y,m], time=self.times[x,y,m], dist=self.dists[x,y,m]))
        if user_action is not None and self._is_valid(user_action):
            self.path_user = []
            for x,y,m in user_action:
                self.path_user.append(Route(x, y, m, price=self.prices[x,y,m], time=self.times[x,y,m], dist=self.dists[x,y,m]))

    def _is_valid(self, path):
        valid = True
        for x,y,m in path:
            if self.bin_edges[x][y][m] == "0":
                valid = False
                break
        if not valid:
            print("Invalid pathway!")
        return valid

    def display_path(self):
        plt.figure(figsize=(14,12))
        ax = plt.gca()
        ax.set_title("Current Path")
        G = nx.Graph()
        nx.draw_networkx_nodes(G, self.nodes_pos, nodelist=[str(i) for i in range(self.n_cities)],
                               node_color="tab:red", node_size=800, alpha=0.9)
        if self.path_ai is not None:
            nx.draw_networkx_edges(G, self.nodes_pos, edgelist=[(str(route.start), str(route.end)) for route in self.path_ai], width=7, edge_color='blue', alpha=0.5)
            nx.draw_networkx_edge_labels(G, self.nodes_pos, edge_labels={(str(route.start), str(route.end)):"mode: "+str(route.mode) for route in self.path_ai},
                                         label_pos=0.6, font_size=9, font_color='blue', alpha=0.8, horizontalalignment='center', verticalalignment='bottom')
        if self.path_user is not None:
            nx.draw_networkx_edges(G, self.nodes_pos, edgelist=[(str(route.start), str(route.end)) for route in self.path_user], width=7, edge_color='green', alpha=0.6)
            nx.draw_networkx_edge_labels(G, self.nodes_pos, edge_labels={(str(route.start), str(route.end)):"mode: "+str(route.mode) for route in self.path_user},
                                         label_pos=0.4, font_size=9, font_color='green', alpha=0.8, horizontalalignment='center', verticalalignment='top')
        nx.draw_networkx_labels(G, self.nodes_pos, {str(i):str(i) for i in range(self.n_cities)}, font_size=20, font_weight="bold", font_color="whitesmoke")
        nx.draw_networkx_nodes(G, self.nodes_pos, nodelist=[str(self.start),str(self.destination)], node_color="purple")
        _ = ax.axis('off')
        plt.show()

    def display(self):
        n_rows = int(np.ceil(self.n_modes/3))
        plt.figure(figsize=(21, 6*n_rows))
        for i in range(self.n_modes):
            plt.subplot(n_rows, 3, i+1)
            self.display_mode(i)
        plt.show()

    def display_mode(self, mode):
        #plt.figure(figsize=(12,9))
        ax = plt.gca()
        ax.set_title("Transport mode: " + str(mode))
        G = nx.Graph()
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if self.bin_edges[i][j][mode] == "1":
                    G.add_edge(str(i), str(j))
        nx.draw(G, self.nodes_pos, ax=ax, with_labels=True, width=3, edge_color='grey', font_color="whitesmoke",
                font_weight="bold", font_size=20, node_color="tab:red", node_size=800, alpha=0.9)
        nx.draw_networkx_nodes(G, self.nodes_pos, nodelist=[str(self.start),str(self.destination)], node_color="purple")
        _ = ax.axis('off')
        #plt.show()

    def _find_best_pos(self):
        plt.figure(figsize=(7,6))
        G = nx.Graph()
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if "1" in self.bin_edges[i][j]:
                    G.add_edge(str(i), str(j))
        self.nodes_pos = nx.kamada_kawai_layout(G)
        plt.show("off")
        

    def _encrypt_route(self, start, end):
        return str(start)+"#"+str(end)

    def _decrypt_rout(self, key):
        start, end = key.split("#")
        return int(start), int(end)

    def _generate_random_map(self):
        edges_dec = np.zeros((self.n_cities, self.n_cities)).astype(np.int16)
        for i in range(self.n_cities):
            for j in range(i):
                if np.random.random() < self.modes_prob:
                    edges_dec[i][j] = edges_dec[j][i] = np.random.randint(2**self.n_modes)
        self.bin_edges = [[dec2bin(edge, self.n_modes) for edge in row] for row in edges_dec]

        # pass information to graph
        self.graph.add_nodes_from(range(1,self.n_cities+1))
        for c,city in enumerate(self.bin_edges):
            mode_mask = ['1' in mode for mode in city]
            edge_ind = [i for i,x in enumerate(mode_mask) if x]
            self.graph.add_edges_from([(c,ind) for ind in edge_ind])
    
    def _generate_properties(self, modes_args):
        props = np.zeros((self.n_cities, self.n_cities, self.n_modes))
        for i in range(self.n_cities):
            for j in range(i):
                for m in range(self.n_modes):
                    if self.bin_edges[i][j][m] == "1":
                        props[i][j][m] = props[j][i][m] = np.random.normal(*modes_args[m])
                    else:
                        props[i][j][m] = props[j][i][m] = np.inf
        return props

class User:
    def __init__(self,**kwargs):
        self.world_setup = None
        self.user_params = kwargs.get('user_param', np.random.normal(size=(1,3)))
        self.error = np.random.uniform(0,.3)
        self.observations = []


    def take_action(self):
        assert len(self.observations) > 0, "observe() needs to be called first"
        last_ai = self.observations[-1]
        ai_advice = [r.get_itinerary() for r in last_ai]

        actions, action_prob = self.user_policy(ai_advice)
        sampled_action = np.random.choice(np.array(actions, dtype=object), 
                                          p=action_prob)
        return sampled_action
        
        
    def user_policy(self, ai_advice):
        segment_costs = [self.calc_segment_cost(segment) for segment in ai_advice]

        # COULD DO: make Boltzmann ration as well
        most_costly = np.argmax(segment_costs)
        alt_dict = self._find_alternatives(ai_advice,most_costly)

        all_paths,cost_vec = [],[]
        for journey, cost in alt_dict.items():
            new_path = copy.copy(ai_advice)
            new_costs = copy.copy(segment_costs)
            if len(journey)==3:
                new_path[most_costly] = journey
                new_costs[most_costly] = cost
                
            elif len(journey)==2:
                part1, part2 = journey
                new_path[most_costly] = part1
                new_path.insert(most_costly+1,part2)
                new_costs[most_costly] = cost
            all_paths.append(new_path)
            cost_vec.append(np.sum(new_costs))

        utility = -1*np.array(cost_vec) # Maybe students need to actually call the utility + all possible actions
        policy_space = np.exp(utility)/np.sum(np.exp(utility)) # Have students implement this?

        # COULD DO: In case we need more stochasticity
        # make_error = bernoulli(self.error)
        # if make_error:
        #     new_path_dict.drop(preferred_path)
        #     return np.random.choice(list(new_path_dict.values()))
        return all_paths, policy_space
        

    def calc_segment_cost(self,segment):
        if isinstance(segment,Route):
            cost_vec = segment.get_costs()
        else:
            cost_vec = np.array([self.world_setup['dists'][segment], 
                              self.world_setup['prices'][segment],
                              self.world_setup['times'][segment]])
        return self.user_params.dot(cost_vec)

    
    def _find_alternatives(self,path,prob_ind):
        alt_dict = dict()
        start, mid_pt, mode = path[prob_ind]
        
        # change mode of transport
        bin_modes = self.world_setup['edges'][start, mid_pt]
        alt_modes = [i for i, val in enumerate(bin_modes) if val=='1']
        for mode_num in alt_modes:
            new_mode = (start,mid_pt,mode_num)
            alt_dict[new_mode] = self.calc_segment_cost(new_mode)
        
        if path[prob_ind] == path[-1]:
            return alt_dict
        
        # find alternative transfer routes
        _, end, mode = path[prob_ind+1]
        ## skip
        bin_direct = self.world_setup['edges'][start, end]
        alt_direct = [i for i, val in enumerate(bin_direct) if val=='1']
        for mode_num in alt_direct:
            new_direct = (start,end,mode_num)
            alt_dict[new_direct] = self.calc_segment_cost(new_direct)

        ## change transfer
        alt_transfer1 = self.world_setup['edges'][start, :]
        for mid_pt, bin_transfer1 in enumerate(alt_transfer1):
            txn_dict = dict()
            alt_tnx1 = [i for i, v in enumerate(bin_transfer1) if v=='1']
            for mode_num in alt_tnx1:
                txn1 = (start,mid_pt,mode_num)
                txn_dict[txn1] = self.calc_segment_cost(txn1)
            bin_tnx2 = self.world_setup['edges'][mid_pt, end]
            alt_tnx2 = [i for i, v in enumerate(bin_tnx2) if v=='1']
            for mode_num in alt_tnx2:
              txn2 = (mid_pt,end,mode_num)
              cost2 = self.calc_segment_cost(txn2)
              for txn1,cost1 in txn_dict.items():
                  alt_dict[(txn1,txn2)] = cost1 + cost2

        return alt_dict


    def observe(self,state):
        # If first time seeing the board
        if self.world_setup is None:
            self.world_setup = dict()
            self.world_setup['edges'] = np.array(state.bin_edges)
            self.world_setup['prices'] = state.prices
            self.world_setup['times'] = state.times
            self.world_setup['dists'] = state.dists
        if state.path_ai is not None:
            self.observations.append(state.path_ai)