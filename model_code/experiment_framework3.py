import fast_model_improved as fmi
from pyexpat import model
import copy

def choose_influencers(model, num_influencers, method='random'):
    if method == 'random':
        return model.random_influencers(num_influencers)
    elif method == 'hill_climbing':
        return model.greedy_hill_climbing(num_influencers, steps=10, R=10)
    else:
        raise ValueError("Unsupported method for selecting influencers")

def run_influence_cascade(graph, num_influencers, steps, selection_method='random'):
    # Initialize the model
    model = fmi.InfluenceDeinfluenceModel(graph, selection_method)
    # Choose influencers
    influencers = choose_influencers(model, num_influencers, method=selection_method)
    model.set_influencers(influencers)
    model.selected_influencers = influencers
    # Run the cascade
    model.run_cascade(steps)
    # Return the updated graph and model
    return model

def run_simple_cascade(steps):
    model.set_influencers(model.selected_influencers)
    model.run_cascade(steps)
    return model

def select_deinfluencers(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['RdExIniInf'] = model.select_deinfluencers_from_not_ini_influencers(k)
        deinfluencers_dict['RdExAllInf'] = model.select_deinfluencers_from_not_influencers(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['Closeness'] = model.select_deinfluencers_closeness_centrality(k)
        deinfluencers_dict['Betweenness'] = model.select_deinfluencers_betweenness_centrality(k)
        deinfluencers_dict['Eigenvector'] = model.select_deinfluencers_eigenvector_centrality(k, max_iter=1000, tol=1e-06)
        deinfluencers_dict['PageRank'] = model.select_deinfluencers_pagerank_centrality(k)
        deinfluencers_dict['RdIniInf'] = model.select_deinfluencers_from_ini_influencers(k)
        deinfluencers_dict['RdAllInf'] = model.select_deinfluencers_from_influencers(k)
        deinfluencers_dict['RRkIniInf'] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
        deinfluencers_dict['RRkAllInf'] = model.select_deinfluencers_from_influencers_degree_centrality(k)
        
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def shuffle_deinfluencers(model, k, deinfluencers_dict):
    methods_to_shuffle = ['Random', 'RdExIniInf', 'RdExAllInf', 'RdIniInf', 'RdAllInf', 'RRkIniInf', 'RRkAllInf']
    for method in methods_to_shuffle:
        if method in deinfluencers_dict:
            if method == 'Random':
                deinfluencers_dict[method] = model.select_deinfluencers_random(k)
            elif method == 'RdExIniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_not_ini_influencers(k)
            elif method == 'RdExAllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_not_influencers(k)
            elif method == 'RdIniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_ini_influencers(k)
            elif method == 'RdAllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_influencers(k)
            elif method == 'RRkIniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
            elif method == 'RRkAllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_influencers_degree_centrality(k)
    return deinfluencers_dict

# Define the combined count function
def count_deinfluenced(model, deinfluencers, num_runs, steps):
    total_deinfluenced = 0
    total_influenced = 0
    total_transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}
    # Create a deep copy of the model to ensure initial influencers remain the same
    initial_model = copy.deepcopy(model)
    
    for _ in range(num_runs):
        # Reset the model to the initial state with the same influencers
        model = copy.deepcopy(initial_model)
        model.reset_transition_counts()
        model.set_deinfluencers(deinfluencers)
        model.run_cascade(steps)
        
        total_deinfluenced += model.evaluate_deinfluence()
        total_influenced += model.evaluate_influence()
        
        for key in total_transition_counts:
            total_transition_counts[key] += model.transition_counts[key]
            
    return total_deinfluenced / num_runs, total_influenced / num_runs, {key: total / num_runs for key, total in total_transition_counts.items()}

def average_results(deinfluencers_list, model, num_runs, steps):
    cumulative_results = {}

    for k, deinfluencers_methods in deinfluencers_list:
        if k not in cumulative_results:
            cumulative_results[k] = {method: (0, 0, {'I->S': 0, 'D->S': 0, 'D->I': 0}) for method in deinfluencers_methods.keys()}
        
        for _ in range(num_runs):
            shuffled_deinfluencers_methods = {method: shuffle_deinfluencers(model, k, deinfluencers) if method in ['Random', 'RdExIniInf', 'RdExAllInf', 'RdIniInf', 'RdAllInf', 'RRkIniInf', 'RRkAllInf'] else deinfluencers for method, deinfluencers in deinfluencers_methods.items()}
            results = {
                method: count_deinfluenced(model, deinfluencers, num_runs, steps)
                for method, deinfluencers in shuffled_deinfluencers_methods.items()
            }
            
            for method, result in results.items():
                cumulative_results[k][method] = (
                    cumulative_results[k][method][0] + result[0],
                    cumulative_results[k][method][1] + result[1],
                    {key: cumulative_results[k][method][2][key] + result[2][key] for key in result[2]}
                )
    
    average_results = {
        k: {
            method: (
                cumulative_results[k][method][0] / num_runs,
                cumulative_results[k][method][1] / num_runs,
                {key: cumulative_results[k][method][2][key] / num_runs for key in cumulative_results[k][method][2]}
            )
            for method in cumulative_results[k]
        }
        for k in cumulative_results
    }
    
    return average_results

def average_results_without_shuffle(deinfluencers_list, model, num_runs, steps):
    cumulative_results = {}

    for k, deinfluencers_methods in deinfluencers_list:
        if k not in cumulative_results:
            cumulative_results[k] = {method: (0, 0, {'I->S': 0, 'D->S': 0, 'D->I': 0}) for method in deinfluencers_methods.keys()}
        
        for _ in range(num_runs):
            results = {
                method: count_deinfluenced(model, deinfluencers, num_runs, steps)
                for method, deinfluencers in deinfluencers_methods.items()
            }
            
            for method, result in results.items():
                cumulative_results[k][method] = (
                    cumulative_results[k][method][0] + result[0],
                    cumulative_results[k] [method][1] + result[1],
                    {key: cumulative_results[k][method][2][key] + result[2][key] for key in result[2]}
                )
    
    average_results = {
        k: {
            method: (
                cumulative_results[k][method][0] / num_runs,
                cumulative_results[k][method][1] / num_runs,
                {key: cumulative_results[k][method][2][key] / num_runs for key in cumulative_results[k][method][2]}
            )
            for method in cumulative_results[k]
        }
        for k in cumulative_results
    }
    
    return average_results
