import networkx as nx
import random
import matplotlib.pyplot as plt
from copy import deepcopy

class InfluenceDeinfluenceModel:
    def __init__(self, graph):
        self.graph = graph
        self.init_edge_weights()
        self.set_initial_states()
        self.history = []  # To store the history of node states

    def init_edge_weights(self):
        for u, v in self.graph.edges:
            p_is = random.uniform(0, 1)
            p_ds = random.uniform(0, 1)
            p_di = 1 - (1 - p_is)**1.5  # c is chosen to be 1.5 for demonstration
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

    def spread_influence(self):
        new_influenced = set()
        new_deinfluenced = set()

        nodes = list(self.graph.nodes)
        random.shuffle(nodes)  # Shuffle the nodes to process them in a random order

        for node in nodes:
            if self.graph.nodes[node]['state'] == 'I':
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['state'] == 'S' and random.random() < self.graph[node][neighbor]['p_is']:
                        new_influenced.add(neighbor)
            elif self.graph.nodes[node]['state'] == 'D':
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['state'] == 'S' and random.random() < self.graph[node][neighbor]['p_ds']:
                        new_deinfluenced.add(neighbor)
                    elif self.graph.nodes[neighbor]['state'] == 'I' and random.random() < self.graph[node][neighbor]['p_di']:
                        new_deinfluenced.add(neighbor)

        for node in new_influenced:
            self.graph.nodes[node]['state'] = 'I'
        for node in new_deinfluenced:
            self.graph.nodes[node]['state'] = 'D'

        self.store_history()  # Store the state of nodes after each step
    
    def influencer_spread_influence(self):
        new_influenced = set()

        for node, data in self.graph.nodes(data=True):
            if data['state'] == 'I':
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['state'] == 'S' and random.random() < self.graph[node][neighbor]['p_is']:
                        new_influenced.add(neighbor)

        for node in new_influenced:
            self.graph.nodes[node]['state'] = 'I'

    def store_history(self):
        # Store a copy of the current state of all nodes
        current_state = {node: self.graph.nodes[node]['state'] for node in self.graph.nodes}
        self.history.append(current_state)

    def run_cascade(self, steps):
        self.store_history()  # Store the initial state
        for _ in range(steps):
            self.spread_influence()

    def run_cascade_influencer(self, steps):
        for _ in range(steps):
            self.influencer_spread_influence()

    def get_history(self):
        return self.history

    def display_graph(self, step):
        if step < 0 or step >= len(self.history):
            print(f"Step {step} is out of range")
            return

        current_state = self.history[step]
        color_map = ['red' if current_state[node] == 'I' else 'blue' if current_state[node] == 'D' else 'lightgrey' for node in self.graph.nodes]

        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8), dpi=200)  # Increase the figure size and DPI
        nx.draw(self.graph, pos, with_labels=True, node_color=color_map, arrows=True)
        edge_labels = {(u, v): f"{self.graph[u][v]['p_is']:.2f}\n{self.graph[u][v]['p_ds']:.2f}\n{self.graph[u][v]['p_di']:.2f}" for u, v in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=5)
        plt.title(f'Step {step}')
        plt.show()

    def display_graphs_grid(self):
        steps = len(self.history)
        cols = 3
        rows = (steps + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axs = axs.flatten()

        pos = nx.spring_layout(self.graph)
        for i in range(steps):
            ax = axs[i]
            current_state = self.history[i]
            color_map = ['red' if current_state[node] == 'I' else 'blue' if current_state[node] == 'D' else 'lightgrey' for node in self.graph.nodes]

            nx.draw(self.graph, pos, ax=ax, with_labels=True, node_color=color_map, arrows=True)
            edge_labels = {(u, v): f"{self.graph[u][v]['p_is']:.2f}\n{self.graph[u][v]['p_ds']:.2f}\n{self.graph[u][v]['p_di']:.2f}" for u, v in self.graph.edges}
            #nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
            ax.set_title(f'Step {i}')

        for j in range(steps, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def evaluate_influence(self):
        """Evaluate the number of influenced nodes."""
        influenced_count = sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I')
        return influenced_count
    
    def greedy_hill_climbing(self, k, steps, R=1000):
        """Select k initial influencers using the improved greedy algorithm."""
        best_influencers = set()

        for _ in range(k):
            best_candidate = None
            best_score = -1

            for node in self.graph.nodes:
                if node in best_influencers:
                    continue
                
                # Temporarily add the candidate node to the set of influencers
                current_influencers = best_influencers | {node}
                total_score = 0

                for _ in range(R):
                    self.set_initial_states()
                    self.set_influencers(current_influencers)
                    self.run_cascade_influencer(steps)
                    total_score += self.evaluate_influence()

                avg_score = total_score / R

                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                best_influencers.add(best_candidate)

        return best_influencers
    
    # New Greedy Algorithm
    
    def generate_random_graph(self, p):
        random_graph = nx.DiGraph()
        for u, v in self.graph.edges:
            if random.random() < p:
                random_graph.add_edge(u, v)
        print("Random Graph Edges:", random_graph.edges())  # Debugging line
        return random_graph

    def compute_reachable_set(self, graph, seeds):
        reachable = set(seeds)
        to_explore = list(seeds)
        while to_explore:
            current = to_explore.pop()
            if current in graph:  # Ensure current node exists in the graph
                for neighbor in graph.neighbors(current):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        to_explore.append(neighbor)
        print("Reachable Set for Seeds {}: {}".format(seeds, reachable))  # Debugging line
        return reachable

    def greedy_hill_climbing_new(self, k, R=20000, p=0.1):
        """Select k initial influencers using the NewGreedyIC algorithm."""
        S = []
        for _ in range(k):
            sv = {v: 0 for v in self.graph if v not in S}
            for _ in range(R):
                G_prime = self.generate_random_graph(p)
                reachable_S = self.compute_reachable_set(G_prime, S)
                for v in sv:
                    if v not in reachable_S:
                        reachable_v = self.compute_reachable_set(G_prime, {v})
                        sv[v] += len(reachable_v)
            for v in sv:
                sv[v] /= R
            best_candidate = max(sv, key=sv.get)
            S.append(best_candidate)
        return S
