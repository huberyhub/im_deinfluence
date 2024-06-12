from joblib import Parallel, delayed
import networkx as nx
import random

class InfluenceDeinfluenceModel:
    def __init__(self, graph):
        self.graph = graph
        self.init_edge_weights()
        self.activated_edges = set()

    def init_edge_weights(self):
        for u, v in self.graph.edges:
            p_is = random.uniform(0, 1)
            p_ds = random.uniform(0, 1)
            p_di = random.uniform(0, 1)
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

        for edge in self.active_edges:
            node, neighbor = edge
            if self.graph.nodes[node]['state'] == 'I' and self.graph.nodes[neighbor]['state'] == 'S':
                new_influenced.add(neighbor)
            elif self.graph.nodes[node]['state'] == 'D':
                if self.graph.nodes[neighbor]['state'] == 'S':
                    new_deinfluenced.add(neighbor)
                elif self.graph.nodes[neighbor]['state'] == 'I':
                    new_deinfluenced.add(neighbor)

        for node in new_influenced:
            self.graph.nodes[node]['state'] = 'I'
        for node in new_deinfluenced:
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
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.spread_influence()

    def run_cascade_influencer(self, steps):
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.influencer_spread_influence()

    def evaluate_influence(self):
        """Evaluate the number of influenced nodes."""
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I')

    def greedy_hill_climbing(self, k, steps, R=10):
        #Select k initial influencers using the improved greedy algorithm.
        best_influencers = set()

        def simulate_influence(current_influencers):
            total_score = 0
            for _ in range(R):
                self.reset_graph()
                self.set_influencers(current_influencers)
                self.run_cascade_influencer(steps)
                #print("Influence:", self.evaluate_influence())
                total_score += self.evaluate_influence()
            return total_score / R

        for _ in range(k):
            best_candidate = None
            best_score = -1

            candidates = [node for node in self.graph.nodes if node not in best_influencers]
            scores = Parallel(n_jobs=-1)(delayed(simulate_influence)(best_influencers | {node}) for node in candidates)

            best_score, best_candidate = max(zip(scores, candidates))

            if best_candidate is not None:
                best_influencers.add(best_candidate)

        return best_influencers

    def reset_graph(self):
        self.set_initial_states()
        self.activated_edges = set()