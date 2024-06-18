from joblib import Parallel, delayed
import networkx as nx
import random

class InfluenceDeinfluenceModel:
    """
    A class representing an influence-deinfluence model for spreading influence in a graph.

    Attributes:
        graph (networkx.Graph): The graph on which the influence-deinfluence model operates.
        activated_edges (set): A set of activated edges in the graph.
        selected_influencers (set): A set of selected influencers in the graph.

    Methods:
        __init__(self, graph, edeg_weights_type='random'): Initializes the InfluenceDeinfluenceModel object.
        edge_weights(self, type): Sets the edge weights based on the specified type.
        random_edge_weights(self): Sets random edge weights for the graph.
        fixed_edge_weights(self, p_is, p_ds, p_di): Sets fixed edge weights for the graph.
        dominate_edge_weights(self, c): Sets dominate edge weights for the graph.
        set_initial_states(self): Sets the initial states of the nodes in the graph.
        set_influencers(self, influencers): Sets the influencers in the graph.
        set_deinfluencers(self, deinfluencers): Sets the deinfluencers in the graph.
        pre_determine_active_edges(self): Determines the active edges based on the current states of the nodes.
        spread_influence(self): Spreads influence in the graph based on the active edges.
        influencer_spread_influence(self): Spreads influence only from influencers in the graph.
        run_cascade(self, steps): Runs the influence-deinfluence cascade for the specified number of steps.
        run_cascade_influencer(self, steps): Runs the influence cascade only from influencers for the specified number of steps.
        evaluate_influence(self): Evaluates the number of influenced nodes in the graph.
        greedy_hill_climbing(self, k, steps, R=10): Performs greedy hill climbing to select the best influencers.
        reset_graph(self): Resets the graph to its initial state.
    """

    def __init__(self, graph, edeg_weights_type='random'):
        """
        Initializes the InfluenceDeinfluenceModel object.

        Args:
            graph (networkx.Graph): The graph on which the influence-deinfluence model operates.
            edeg_weights_type (str, optional): The type of edge weights to use. Defaults to 'random'.
        """
        self.graph = graph
        self.edge_weights(edeg_weights_type)
        self.set_initial_states()
        self.activated_edges = set()
        self.selected_influencers = set()

    def edge_weights(self, type):
        if type == 'random':
            self.random_edge_weights()
        elif type == 'fixed':
            self.fixed_edge_weights(p_is=1, p_ds=1, p_di=1)
        elif type == 'dominate':
            self.dominate_edge_weights(c=1)
        else:
            print("Invalid edge weights type. Using random edge weights.")
            self.random_edge_weights()

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
                    if (self.graph.nodes[neighbor]['state'] == 'S' and random.random() < self.graph[node][neighbor]['p_is']):
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
    # Select k initial influencers using the improved greedy algorithm.
        best_influencers = set()

        def simulate_influence(current_influencers):
            total_score = 0
            for _ in range(R):
                self.reset_graph()
                self.set_influencers(current_influencers)
                self.run_cascade_influencer(steps)
                total_score += self.evaluate_influence()
            return (total_score / R)

        for _ in range(k):
            best_candidate = None
            best_score = -1

            candidates = [node for node in self.graph.nodes if node not in best_influencers]
            scores = Parallel(n_jobs=-1)(delayed(simulate_influence)(best_influencers | {node}) for node in candidates)

            best_score, best_candidate = max(zip(scores, candidates), key=lambda x: x[0])

            if best_candidate is not None:
                best_influencers.add(best_candidate)

        self.selected_influencers = best_influencers

        return best_influencers

    def reset_graph(self):
        self.set_initial_states()
        self.activated_edges = set()