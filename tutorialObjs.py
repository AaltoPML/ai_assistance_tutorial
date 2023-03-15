import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, multivariate_normal
from scipy.optimize import minimize
import copy, time

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
    def __init__(self, n_cities=10, n_modes=5, modes_prob=.6, 
                  modes_dist=None, modes_price=None, modes_time=None):
        np.random.seed(12345)
        
        self.n_cities = n_cities
        self.n_modes = n_modes
        self.modes_prob = modes_prob
        self.bin_edges = None
        self.graph = nx.Graph()
        self._generate_random_map()
        self.start, self.destination = np.random.choice(self.n_cities, size=(2,), replace=False)
        self.path_ai, self.path_user = None, None
        
        # The higher the mode index, more expensive and faster.
        if modes_dist is None:
            modes_dist = [(1, 0.2) for i in range(n_modes)]
        if modes_price is None:
            modes_price = [(2*i+1, 1) for i in range(n_modes)]
        if modes_time is None:
            modes_time = [(5*(n_modes-i), 5) for i in range(n_modes)]
        
        
        # generate price, duration and other factors for the graph
        self.prices = self._generate_properties(modes_price)
        self.times = self._generate_properties(modes_time)
        self.dists = self._generate_properties(modes_dist)
        self._find_best_pos()
        
        # Undo seed setting/ restart randomness
        t = 1000 * time.time() 
        np.random.seed(int(t) % 2**32)

    def reset(self):
        self.path_ai, self.path_user = None, None
        
    def step(self, ai_action=None, user_action=None):        
        if ai_action is not None and self._is_valid(ai_action):
            self.path_ai = []
            for x,y,m in ai_action:
                self.path_ai.append(Route(x, y, m, price=self.prices[x,y,m], 
                                          time=self.times[x,y,m], dist=self.dists[x,y,m]))
        if user_action is not None and self._is_valid(user_action):
            self.path_user = []
            for x,y,m in user_action:
                self.path_user.append(Route(x, y, m, price=self.prices[x,y,m], 
                                            time=self.times[x,y,m], dist=self.dists[x,y,m]))

    def is_solved(self):
        return (self.path_ai == self.path_user) and (self.path_ai is not None)
    
    def _is_valid(self, path):
        valid = True
        for x,y,m in path:
            if self.bin_edges[x][y][m] == "0":
                valid = False
                break
        if not valid:
            print("Invalid pathway!")
        return valid

    def look_up_cost(self,segment):
        return [self.dists[segment], 
                self.prices[segment],
                self.times[segment]]
    
    def display_path(self):
        ax = plt.gca()
        if self.path_ai is not None:
            ax.set_title("Current Path")
        
        G = nx.Graph()
        color_dict = {0:'red',1:'green',2:'blue',3:'cyan',4:'black'}
        nx.draw_networkx_nodes(G, self.nodes_pos, nodelist=[str(i) for i in range(self.n_cities)],
                               node_color="tab:red", node_size=800, alpha=0.9)
        if self.path_ai is not None:
            nx.draw_networkx_edges(G, self.nodes_pos, 
                                   edgelist=[(str(route.start), str(route.end)) for route in self.path_ai], 
                                   width=3, style='--',
                                   edge_color=[color_dict[route.mode] for route in self.path_ai]) #edge_color='blue',alpha=0.5,
            nx.draw_networkx_edge_labels(G, self.nodes_pos, label_pos=0.6, font_size=9, font_color='black',
                                         edge_labels={(str(route.start), str(route.end)):\
                                                      "mode: "+str(route.mode) for route in self.path_ai},
                                         alpha=0.8, horizontalalignment='center', 
                                         verticalalignment='bottom')
        if self.path_user is not None:
            nx.draw_networkx_edges(G, self.nodes_pos, 
                                   edgelist=[(str(route.start), str(route.end)) for route in self.path_user],
                                   edge_color=[color_dict[route.mode] for route in self.path_ai],
                                   width=6,alpha=0.4) #edge_color='green'
            nx.draw_networkx_edge_labels(G, self.nodes_pos, 
                                         edge_labels={(str(route.start), str(route.end)):"mode: "+str(route.mode) for route in self.path_user},
                                         label_pos=0.4, font_size=9, 
                                         font_color='black', alpha=0.8, 
                                         horizontalalignment='center', 
                                         verticalalignment='top')
        nx.draw_networkx_labels(G, self.nodes_pos, 
                                {str(i):str(i) for i in range(self.n_cities)}, 
                                font_size=20, font_weight="bold", font_color="whitesmoke")
        nx.draw_networkx_nodes(G, self.nodes_pos, 
                               nodelist=[str(self.start),str(self.destination)], 
                               node_color="purple")
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
#         plt.figure(figsize=(7,6))
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

class UserModel:
    def __init__(self,simUser=True,**kwargs):
        self.role = "sim" if simUser else "user"
        
#         new_seed = np.random.randint(10000)
        
        # read in overwrites
        self.param_dist = kwargs.get('distribution', 
                                     multivariate_normal()) #seed=new_seed))
        self.user_params = kwargs.get('user_params', 
                                      self.param_dist.rvs(3))
        self.policy_fn = kwargs.get('policy_fn', None)
        self.inf_fn = kwargs.get('inference', None)
        self.posterior_fn = kwargs.get('posterior_fn', None)
        
        # verify tutorial excercise overwrites
        if self.policy_fn is not None:
            assert callable(self.policy_fn)
        if self.inf_fn is not None:
            assert callable(self.inf_fn)
        
        
        self.world_setup = None
        self.observations = []
            
        if self.role == 'sim':
            self.param_dist = multivariate_normal(mean=self.user_params)
        elif self.role == 'user':
            self.error = np.random.uniform(0,.3)

    def take_action(self,**kwargs):
        assert len(self.observations) > 0, "observe() needs to be called first"
        ai_advice, advice_costs = self.observations[-1]
#         ai_advice = [r.get_itinerary() for r in last_ai]
#         advice_costs = [self._calc_segment_cost(segment) for segment in ai_advice]
        # COULD DO: make Boltzmann ration as well
        all_paths, cost_vec = self._find_alternatives(ai_advice,advice_costs)
       
        if self.policy_fn is not None:
            actions, action_prob = self.policy_fn(all_paths,self.user_params)
        else:
            actions, action_prob = self.policy(all_paths,cost_vec)
        
        if np.isnan(action_prob).any():
            print(actions)
            print(all_paths, cost_vec)
        sampled_ind = np.random.choice(range(len(actions)), 
                                          p=action_prob)
        
        return actions[sampled_ind]
        
    def policy(self,all_actions,cost_vec):
        action_cost = np.array(cost_vec)
        action_probs = np.exp(action_cost)/np.sum(np.exp(action_cost))

        # COULD DO: In case we need more stochasticity
        # make_error = bernoulli(self.error)
        # if make_error:
        #     new_path_dict.drop(preferred_path)
        #     return np.random.choice(list(new_path_dict.values()))
        return all_actions, action_probs
    
    def _calc_segment_cost(self,segment):
        if isinstance(segment,Route):
            cost_vec = segment.get_costs()
        else:
            cost_vec = np.array([self.world_setup['dists'][segment], 
                                 self.world_setup['prices'][segment],
                                 self.world_setup['times'][segment]])
        return self.user_params.dot(cost_vec) 
    
    def _find_alternatives(self,path,path_costs):
        costly_ind = np.argmax(path_costs)
        new_path = copy.copy(path)
        new_costs = copy.copy(path_costs)
        
        
        all_paths, cost_vec = [],[]
        # change mode of transport
        start, end, _ = path[costly_ind]
        bin_modes = self.world_setup['edges'][start, end]
        alt_modes = [i for i, val in enumerate(bin_modes) if val=='1']
        for mode_num in alt_modes:
            new_mode = (start,end,mode_num)
#             if new_mode != path[costly_ind]:
            new_path[costly_ind] = new_mode
            new_costs[costly_ind] = self._calc_segment_cost(new_mode)
            all_paths.append(new_path)
            cost_vec.append(np.sum(new_costs))
            new_path = copy.copy(path)
            new_costs = copy.copy(path_costs)

        # find alternative transfer routes
        ## change transfer
        alt_transfer1 = self.world_setup['edges'][start, :]
        txn_dict = dict()
        for mid_pt, bin_transfer1 in enumerate(alt_transfer1):
            alt_tnx1 = [i for i, v in enumerate(bin_transfer1) if v=='1']
            for mode_num in alt_tnx1:
                txn1 = (start,mid_pt,mode_num)
                if (txn1 == path[costly_ind]):
                    continue
                else:
                    txn_dict[txn1] = self._calc_segment_cost(txn1)

        for part1,cost1 in txn_dict.items():
            _,txn_loc,_ = txn1
            bin_tnx2 = self.world_setup['edges'][txn_loc, end]
            alt_tnx2 = [i for i, v in enumerate(bin_tnx2) if v=='1']
            for mode_num in alt_tnx2:
                part2 = (txn_loc,end,mode_num)
                cost2 = self._calc_segment_cost(part2)

                new_path[costly_ind] = part1
                new_costs[costly_ind] = cost1
                new_path.insert(costly_ind+1,part2)
                new_costs.insert(costly_ind+1,cost2)

                all_paths.append(new_path)
                cost_vec.append(np.sum(new_costs))
                new_path = copy.copy(path)
                new_costs = copy.copy(path_costs)

        ## skip
        if path[costly_ind] != path[-1]:
            _, new_end, _ = path[costly_ind+1]
            ## skip
            bin_direct = self.world_setup['edges'][start, new_end]
            alt_direct = [i for i, val in enumerate(bin_direct) if val=='1']
            for mode_num in alt_direct:
                new_direct = (start,new_end,mode_num)
                new_path[costly_ind] = new_direct
                new_costs[costly_ind] = self._calc_segment_cost(new_direct)
                del new_path[costly_ind+1]
                del new_costs[costly_ind+1]

                all_paths.append(new_path)
                cost_vec.append(np.sum(new_costs))
                new_path = copy.copy(path)
                new_costs = copy.copy(path_costs)

        return all_paths, cost_vec
    
    def observe(self,state):
        # If first time seeing the board
        if self.world_setup is None:
            self.world_setup = dict()
            self.world_setup['edges'] = np.array(state.bin_edges)
            self.world_setup['prices'] = state.prices
            self.world_setup['times'] = state.times
            self.world_setup['dists'] = state.dists
        if state.path_ai is not None:
            ai_advice = [r.get_itinerary() for r in state.path_ai]
            advice_costs = [self._calc_segment_cost(segment) for segment in ai_advice]
            self.observations.append((ai_advice,advice_costs))
          
        # Making this call explicit for tutorial excercise
#         if self.role == 'sim':
#             self.update(state)
            
    def sample(self):
        assert self.role=='sim', "Method only available to simulated users; method called on true user."
        return self.param_dist.rvs(size=1)
    
    def update(self, obs,**kwargs):
        assert self.role=='sim', "Method only available to simulated users; method called on true user."
        
#         if obs.path_ai != obs.path_user:
        if not obs.is_solved():
            # step 1: check what user changed
            ai_advice = [s.get_itinerary() for s in obs.path_ai]
            changes = [s for s in obs.path_user if s.get_itinerary() not in ai_advice]
            for c in changes:
                cost_vec = c.get_costs()
                self.param_dist = self.inference_engine(cost_vec)

                # step 2: Use Laplace Approx for posterior
                # Debug idea: change init value every time
#                 init_mean = self.param_dist.mean
#                 optim = minimize(self._log_posterior, init_mean,
#                                  args=(cost_vec,"L1"), method='BFGS')
#                 if not optim.success:
#                     print("Failed to minimize")
#                 w_map = optim.x/np.sum(optim.x)
#                 hessian = np.linalg.inv(optim.hess_inv)

#                 print(w_map,np.any(hessian<0))
#                 print(hessian)
#                 self.param_dist = multivariate_normal(mean=w_map,
#                                                       cov=hessian)
        self.user_params = self.param_dist.rvs(size=1)
                  
    def inference_engine(self,e):
        if self.inf_fn is not None:
            approx_dist = self.inf_fn(e)
        else:
            # step 2: Use Laplace Approx for posterior
            # Debug idea: change init value every time
            init_mean = self.param_dist.mean
            optim = minimize(self._log_posterior, init_mean,
                             args=(e,"L1"), method='BFGS')
#             while not optim.success:
            if not optim.success:
#                 print("Failed to minimize")
#                 print(optim)
                return multivariate_normal(mean=self.param_dist.mean,
                                              cov=self.param_dist.cov)
                
            w_map = optim.x/np.sum(optim.x)
            hessian = np.linalg.inv(optim.hess_inv) 
            
            # Due to the nature of the exp function, the optimizer sometimes 
            # finds a saddle point. To use the Hessian as a covariance
            # matrix, it needs to be positive definite, so the absolute
            # value is taken.
            approx_dist = multivariate_normal(mean=w_map,
                                              cov=np.abs(hessian))
        
        return approx_dist

        
    
    def _log_posterior(self,w,e,regularizer=None):
        if self.posterior_fn is not None:
            post_pr = self.posterior_fn(w,e)
        else:
            
#         w_mu = self.param_dist.mean
#         w_cov = self.param_dist.cov
#         log_prior = (w - w_mu).dot(np.linalg.inv(w_cov)).dot(w - w_mu)
#         if not np.isclose(np.sum(w),1):
#             w = w/np.sum(w)
            log_prior = self.param_dist.logpdf(w)
            log_likelihood = w.dot(e) 
            post_pr =  -1 * (log_prior + log_likelihood)
        
        if regularizer == "L1":
            l1_reg = np.sum(np.abs(w))
            return -1*(post_pr + 10) #np.log(l1_reg))
#         elif regularizer == "L2":    # Optimization yields invalid Hessian for covar matrix
#             l2_reg = np.sum(np.square(w))
#             return -1*(log_prior + log_likelihood + l2_reg)
        
            
        return post_pr
    

class crUser(UserModel):
    # implement user cross over
    def __init__(self,simUser=False,**kwargs):
        super().__init__(**kwargs)

class Assistant:
    # MLE infer user params
    # plan assuming MLE
    # delta-dirac
    # analytical closed form posterior of the current user param estimate
    # return param sample
    def __init__(self,**kwargs):
        self.user_model = kwargs.get('user_model', UserModel())
        self.strategy = kwargs.get('policy','global_min')
        self.planner = None
        self.env = None
        self.observations = []
    
    def observe(self, obs):
        if self.env is None:
            self.env = Env(obs,self.user_model)
        if obs.path_user is not None:
            self.user_model.update(obs)
            self.env.update(obs)
#             self._update(obs)
            self.observations.append(obs)
    
    def take_action(self):
        best_action = self.policy()
        return best_action
    
    def policy(self):
        if self.strategy is None:
            best_journey = self.planner.recommend(self.env, self.user_model)
        elif self.strategy=="global_min": 
            global_cost = []
            for path in self.env.path_tuples:
                itinerary, cost_vec = self.env.run_scenario(path)
                global_cost.append(np.sum(cost_vec))
#             scenes = [self.env.run_scenario(p) for p in self.env.path_tuples]
            global_min = np.argmin(global_cost)
            min_path = self.env.path_tuples[global_min]
            best_journey,_ = self.env.run_scenario(min_path)
            return best_journey
#             prop_sum = []
#             total_cost = [np.sum()]
#             print(
#             min_ind = np.argmin(np.sum(cost_vec,axis=0))
#             min_path = self.env.path_tuples[min_ind]
#             path, cost = 
#             return self.env.run_scenario(min_path)
                
        else:
            rand_ind = np.random.randint(0,len(self.env.path_tuples))
            random_path = self.env.path_tuples[rand_ind]
            best_journey,_ = self.env.run_scenario(random_path)
#             print(best_journey)
        return best_journey
    
    def reset(self,**kwargs):
        self.user_model = kwargs.get('user_model', UserModel())
        self.strategy = kwargs.get('policy','global_min')
        self.planner = None
        self.env = None
        self.observations = []
#     def _update(self, obs):
#         pass
    
    
class Env:
    def __init__(self,world_state,user_model):
        # Init simulated world
        self.world = copy.copy(world_state)
        self.start = world_state.start
        self.destination = world_state.destination
        self.bin_edges = np.array(world_state.bin_edges)
        self._find_valid_paths()
        # place holder:
#         self._find_path_properties()

        # Init simulated user
        self.user_model = user_model
        self.user_model.observe(self.world)

    def _find_valid_paths(self):
        self.all_paths = all_paths = list(nx.all_simple_paths(self.world.graph, 
                                                  source=self.start, 
                                                  target=self.destination))
        
        # Generate modes for known task connections
        self.modes_dict = dict()
        self.path_tuples = []
        for path in all_paths:
            cur_path = []
            for segment in range(len(path)-1):
                c_begin, c_end = path[segment], path[segment+1]
                cur_path.append((c_begin, c_end))
                if (c_begin,c_end) not in self.modes_dict.keys():
                    # which mode is available?
                    modes_list = self.world.bin_edges[c_begin][c_end]
                    mode_mask = ['1' in mode for mode in modes_list]
                    mode_ind = [i for i,x in enumerate(mode_mask) if x]
                    self.modes_dict[(c_begin,c_end)] = mode_ind
            self.path_tuples.append(cur_path)
                      
#     def _find_path_properties(self):
        
             
#         return properties_dict
#     def _find_all_actions(self):
#         self.all_actions = []
#         for route in self.path_tuples:
#             route_options = []
#             for segment in route:
#                 modes = self.modes_dict[segment]
#                 for mode in modes:
#                     route_options.append()

    def run_scenario(self,path):
#         journey_dict = dict()
        min_cost_journey, journey_cost = [], []
        for segment in path:
            available_modes = self.modes_dict[segment]
            mode_costs = [self.user_model._calc_segment_cost((*segment,mode)) for mode in available_modes]
            min_ind = np.argmin(np.array(mode_costs))
#             journey_dict[(*segment,available_modes[min_ind])] = np.min(np.array(mode_costs))
            min_cost_journey.append((*segment,available_modes[min_ind]))
            journey_cost.append(np.min(np.array(mode_costs)))
        
        return min_cost_journey, journey_cost
#         return journey_dict
    
    def update(self,obs):
        self.world = copy.copy(obs)
#         self.world.step()
#         user_action = obs.path_user
        
#         self.step()
    
    def step(self, action):
        self.world.step(**action) # dict containing user_action or ai_action  
        

#     def getReward(self):
#         # only needed for terminal states
#         raise NotImplementedError()

#     def __eq__(self, other):
#         raise NotImplementedError()