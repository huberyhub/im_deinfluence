import sidmodel.basic_model as basic_model
import sidmodel.greedy_influencer_model as advanced_model
import networkx as nx

# # Example usage with Erdős-Rényi graph
# model_er = generate_and_run_model('erdos_renyi', num_nodes=20, steps=3, edge_prob=0.2)
# display_model_graphs(model_er)

# # Example usage with Watts-Strogatz graph
# model_ws = generate_and_run_model('watts_strogatz', num_nodes=20, steps=3, k=4, p=0.2)
# display_model_graphs(model_ws)

# Example usage with Barabási-Albert graph

# model_ba = basic_model.generate_and_run_model('barabasi_albert', num_nodes=50, steps=5, m=2)
# basic_model.display_model_graphs(model_ba)

# model_ba = advanced_model.generate_and_run_model('barabasi_albert', num_nodes=40, steps=5, m=2)
# advanced_model.display_model_graphs(model_ba)

# Example usage
num_nodes = 100
steps = 5
edge_prob = 0.1
num_influencers = 10

def generate_and_run_model(graph_type, num_nodes, steps, edge_prob=None, k=None, p=None, m=None, num_influencers=3):
    if graph_type == 'erdos_renyi':
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
    elif graph_type == 'watts_strogatz':
        graph = nx.watts_strogatz_graph(num_nodes, k, p).to_directed()
    elif graph_type == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(num_nodes, m).to_directed()
    else:
        raise ValueError("Unsupported graph type. Choose from 'erdos_renyi', 'watts_strogatz', or 'barabasi_albert'.")

    model = advanced_model.InfluenceDeinfluenceModel(graph)
    model.set_initial_states()
    initial_influencers = model.greedy_hill_climbing(num_influencers, steps)
    #initial_influencers = model.greedy_hill_climbing_new(num_influencers)
    print("Optimized Initial Influencers:", initial_influencers)
    
    model.set_initial_states()
    model.set_influencers(initial_influencers)
    model.set_deinfluencers([2])  # Example: Set a fixed deinfluencer
    model.run_cascade(steps)

    return model

def display_model_graphs(model):
    # for step in range(len(model.get_history())):
    #     model.display_graph(step)
    model.display_graphs_grid()
    
model_er = generate_and_run_model('barabasi_albert', num_nodes, steps, edge_prob=edge_prob, num_influencers=num_influencers, m=2)
display_model_graphs(model_er)
