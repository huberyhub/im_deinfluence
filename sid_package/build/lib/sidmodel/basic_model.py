import networkx as nx
import random
import matplotlib.pyplot as plt

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

    def store_history(self):
        # Store a copy of the current state of all nodes
        current_state = {node: self.graph.nodes[node]['state'] for node in self.graph.nodes}
        self.history.append(current_state)

    def run_cascade(self, steps):
        self.store_history()  # Store the initial state
        for _ in range(steps):
            self.spread_influence()

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


def generate_and_run_model(graph_type, num_nodes, steps, edge_prob=None, k=None, p=None, m=None):
    if graph_type == 'erdos_renyi':
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
    elif graph_type == 'watts_strogatz':
        graph = nx.watts_strogatz_graph(num_nodes, k, p).to_directed()
    elif graph_type == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(num_nodes, m).to_directed()
    else:
        raise ValueError("Unsupported graph type. Choose from 'erdos_renyi', 'watts_strogatz', or 'barabasi_albert'.")

    model = InfluenceDeinfluenceModel(graph)
    model.set_initial_states()
    model.set_influencers([0, 1,3])
    model.set_deinfluencers([2])
    model.run_cascade(steps)

    return model

def display_model_graphs(model):
    # for step in range(len(model.get_history())):
    #     model.display_graph(step)
    model.display_graphs_grid()

