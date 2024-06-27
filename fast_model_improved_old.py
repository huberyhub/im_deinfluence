from joblib import Parallel, delayed
import networkx as nx
import random

class InfluenceDeinfluenceModel:
    def __init__(self, graph, edge_weights_type='random', c=1, p=0.5):
        self.graph = graph
        self.edge_weights(edge_weights_type, c)
        self.set_initial_states()
        self.activated_edges = set()
        self.selected_influencers = set()
        self.selected_deinfluencers = set()
        self.transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}  # Trackers for transitions
        self.p = p  # Probability parameter for resolving simultaneous influence

    def edge_weights(self, type, c):
        if type == 'random':
            self.random_edge_weights()
        elif type == 'fixed':
            self.fixed_edge_weights(p_is=1, p_ds=1, p_di=1)
        elif type == 'dominate':
            self.dominate_edge_weights(c)
        else:
            print("Invalid edge weights type. Using random edge weights.")
            self.random_edge_weights()
    
    def reset_transition_counts(self):
        self.transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}

    def random_edge_weights(self):
        for u, v in self.graph.edges:
            p_is = random.uniform(0, 1)
            p_ds = random.uniform(0, 1)
            p_di = random.uniform(0, 1)
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di

    def fixed_edge_weights(self, p_is, p_ds, p_di):
        for u, v in self.graph.edges:
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di
    
    def dominate_edge_weights(self, c):
        for u, v in self.graph.edges:
            p_is = random.uniform(0, 1)
            p_ds = 1 - (1 - p_is)**c
            p_di = 1 - (1 - p_is)**c
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di

    def set_initial_states(self):
        nx.set_node_attributes(self.graph, 'S', 'state')

    def set_influencers(self, influencers):
        for node in influencers:
            self.graph.nodes[node]['state'] = 'I'

    def set_deinfluencers(self, deinfluencers):
        for node in deinfluencers:
            self.graph.nodes[node]['state'] = 'D'

    def pre_determine_active_edges(self):
        self.active_edges = set()
        for node in self.graph.nodes:
            if self.graph.nodes[node]['state'] == 'I':
                for neighbor in self.graph.neighbors(node):
                    if (self.graph.nodes[neighbor]['state'] == 'S' and 
                        random.random() < self.graph[node][neighbor]['p_is']):
                        self.active_edges.add((node, neighbor))
            elif self.graph.nodes[node]['state'] == 'D':
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['state'] == 'S' and random.random() < self.graph[node][neighbor]['p_ds']:
                        self.active_edges.add((node, neighbor))
                    elif self.graph.nodes[neighbor]['state'] == 'I' and random.random() < self.graph[node][neighbor]['p_di']:
                        self.active_edges.add((node, neighbor))

    def spread_influence(self):
        new_influenced = set()
        new_deinfluenced = set()
        simultaneous_influence = set()

        for edge in self.active_edges:
            node, neighbor = edge
            if self.graph.nodes[node]['state'] == 'I' and self.graph.nodes[neighbor]['state'] == 'S':
                if neighbor in new_deinfluenced:
                    simultaneous_influence.add(neighbor)
                else:
                    new_influenced.add(neighbor)
                    self.transition_counts['I->S'] += 1
            elif self.graph.nodes[node]['state'] == 'D':
                if self.graph.nodes[neighbor]['state'] == 'S':
                    if neighbor in new_influenced:
                        simultaneous_influence.add(neighbor)
                    else:
                        new_deinfluenced.add(neighbor)
                        self.transition_counts['D->S'] += 1
                elif self.graph.nodes[neighbor]['state'] == 'I':
                    new_deinfluenced.add(neighbor)
                    self.transition_counts['D->I'] += 1

        for node in new_influenced:
            if node not in simultaneous_influence:
                self.graph.nodes[node]['state'] = 'I'

        for node in new_deinfluenced:
            if node not in simultaneous_influence:
                self.graph.nodes[node]['state'] = 'D'

        for node in simultaneous_influence:
            # Resolve conflict using parameter p
            if random.random() < self.p:
                self.graph.nodes[node]['state'] = 'I'
            else:
                self.graph.nodes[node]['state'] = 'D'

    def influencer_spread_influence(self):
        new_influenced = set()
        for edge in self.active_edges:
            node, neighbor = edge
            if self.graph.nodes[node]['state'] == 'I' and self.graph.nodes[neighbor]['state'] == 'S':
                new_influenced.add(neighbor)

        for node in new_influenced:
            self.graph.nodes[node]['state'] = 'I'

    def run_cascade(self, steps):
        #self.pre_determine_active_edges()
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.spread_influence()

    def run_cascade_influencer(self, steps):
        #self.pre_determine_active_edges()
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.influencer_spread_influence()

    def evaluate_influence(self):
        """Evaluate the number of influenced nodes."""
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I')
    
    def evaluate_deinfluence(self):
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'D')
    
    def evaluate_susceptible(self):
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'S')
        
    def greedy_hill_climbing(self, k, steps, R=10):
        # Select k initial influencers using the improved greedy algorithm.
        best_influencers = set()

        def simulate_influence(current_influencers):
            total_score = 0
            for _ in range(R):
                self.reset_graph()
                self.set_influencers(current_influencers)
                self.run_cascade_influencer(steps)
                total_score += self.evaluate_influence()
            return total_score / R

        for _ in range(k):
            best_candidate = None
            best_score = -float('inf')

            candidates = [node for node in self.graph.nodes if node not in best_influencers]
            random.shuffle(candidates)  # Shuffle the candidates to avoid any sequence

            scores = Parallel(n_jobs=-1)(delayed(simulate_influence)(best_influencers | {node}) for node in candidates)

            best_score, best_candidate = max(zip(scores, candidates), key=lambda x: x[0])

            if best_candidate is not None:
                best_influencers.add(best_candidate)

        self.selected_influencers = best_influencers

        return best_influencers
    
    def random_influencers(self, k):
        return set(random.sample(list(self.graph.nodes), k))
    
    def random_deinfluencers(self, k):
        return set(random.sample(list(self.graph.nodes), k))

    def reset_graph(self):
        self.set_initial_states()
        self.activated_edges = set()

    def select_deinfluencers_degree_centrality(self, k):
        """Select k deinfluencers based on degree centrality."""
        centrality = nx.degree_centrality(self.graph)
        return sorted(centrality, key=centrality.get, reverse=True)[:k]

    def select_deinfluencers_closeness_centrality(self, k):
        """Select k deinfluencers based on closeness centrality."""
        centrality = nx.closeness_centrality(self.graph)
        return sorted(centrality, key=centrality.get, reverse=True)[:k]

    def select_deinfluencers_betweenness_centrality(self, k):
        """Select k deinfluencers based on betweenness centrality."""
        centrality = nx.betweenness_centrality(self.graph)
        return sorted(centrality, key=centrality.get, reverse=True)[:k]

    def select_deinfluencers_eigenvector_centrality(self, k, max_iter=1000, tol=1e-06):
        """Select k deinfluencers based on eigenvector centrality."""
        try:
            centrality = nx.eigenvector_centrality(self.graph, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            print(f"Power iteration failed to converge within {max_iter} iterations")
            return []
        
        return sorted(centrality, key=centrality.get, reverse=True)[:k]

    def select_deinfluencers_pagerank_centrality(self, k):
        """Select k deinfluencers based on pagerank centrality."""
        centrality = nx.pagerank(self.graph)
        return sorted(centrality, key=centrality.get, reverse=True)[:k]
    
    def select_deinfluencers_random(self, k):
        """Select k random deinfluencers."""
        population = sorted(self.graph.nodes)
        return random.sample(population, k)
    
    def greedy_hill_climbing_deinf(self, j, steps, R=10):
        """Select j de-influencers using greedy algorithm."""
        optimized_deinfluencer = set()
        self.reset_graph()

        for _ in range(j):
            best_candidate = None
            best_score = -1

            for node in self.graph.nodes:
                if node in optimized_deinfluencer:
                    continue
                
                # Temporarily add the candidate node to the set of influencers
                current_deinfluencers = optimized_deinfluencer | {node}
                total_score = 0

                for _ in range(R):
                    self.activated_edges.clear()  # Reset activated edges
                    self.set_initial_states()
                    self.set_influencers(current_deinfluencers)
                    self.run_cascade_influencer(steps)
                    total_score += self.evaluate_influence()

                avg_score = total_score / R

                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                optimized_deinfluencer.add(best_candidate)
            self.reset_graph()

        return optimized_deinfluencer
    
    def select_deinfluencers_from_ini_influencers(self, j):
        influencers = list(self.selected_influencers)  # Convert set to list
        deinfluencers = random.sample(influencers, j)  # Select j deinfluencers randomly from the selected influencers
        return deinfluencers
    
    def select_deinfluencers_from_ini_influencers_degree_centrality(self, j):
        influencers = list(self.selected_influencers)
        return sorted(influencers, key=lambda node: self.graph.degree(node), reverse=True)[:j]
    
    def select_deinfluencers_from_not_ini_influencers(self, j):
        not_influencers = [node for node in self.graph.nodes if node not in self.selected_influencers]
        deinfluencers = random.sample(not_influencers, j)
        return deinfluencers

    def select_deinfluencers_from_influencers(self, j):
        influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I']  # Convert set to list
        deinfluencers = random.sample(influencers, j)  # Select j deinfluencers randomly from the selected influencers
        return deinfluencers
    
    def select_deinfluencers_from_influencers_degree_centrality(self, j):
        influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I']
        return sorted(influencers, key=lambda node: self.graph.degree(node), reverse=True)[:j]
    
    def select_deinfluencers_from_not_influencers(self, j):
        not_influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] != 'I']
        deinfluencers = random.sample(not_influencers, j)
        return deinfluencers