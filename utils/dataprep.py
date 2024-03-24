import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
from sklearn.model_selection import train_test_split
from community import community_louvain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
from scipy.sparse import *
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
import seaborn as sns
np.random.seed(0)


def generate_samples(graph, train_set_ratio):
    """
    Graph pre-processing step required to perform supervised link prediction
    Create training and test sets
    """
        
    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")
       
    
    # --- Step 1: Generate positive edge samples for testing set ---
    residual_g = graph.copy()
    test_pos_samples = []
      
    # Shuffle the list of edges
    edges = list(residual_g.edges())
    np.random.shuffle(edges)
    
    # Define number of positive samples desired
    test_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    train_set_size = graph.number_of_edges() - test_set_size
    num_of_pos_test_samples = 0
    
    # Remove random edges from the graph, leaving it connected
    # Fill in the blanks
    for edge in edges:
        
        # Remove the edge
        residual_g.remove_edge(edge[0], edge[1])
        
        # Add the removed edge to the positive sample list if the network is still connected
        if nx.is_connected(residual_g):
            num_of_pos_test_samples += 1
            test_pos_samples.append(edge)
        # Otherwise, re-add the edge to the network
        else: 
            residual_g.add_edge(edge[0], edge[1])
        
        # If we have collected enough number of edges for testing set, we can terminate the loop
        if num_of_pos_test_samples == test_set_size:
            break

    # Check if we have the desired number of positive samples for testing set 
    if num_of_pos_test_samples != test_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

        
    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = list(residual_g.edges())
        
        
    # --- Step 3: Generate the negative samples for testing and training sets ---
    # Fill in the blanks
    non_edges = list(nx.non_edges(graph))
    np.random.shuffle(non_edges)
    
    train_neg_samples = non_edges[:train_set_size] 
    test_neg_samples = non_edges[train_set_size:train_set_size + test_set_size]
    
    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For testing set
    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1 for _ in test_pos_samples] + [0 for _ in test_neg_samples]
    
    return residual_g, train_samples, train_labels, test_samples, test_labels
     

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)  # Encourages sparsity
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Use if TF-IDF features have been scaled to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

def reduce_dimensions_sparse_tf_idf(node_features):
    # Convert the numpy array to a PyTorch tensor
    features_tensor = torch.FloatTensor(node_features)
    
    # Define the autoencoder and training components
    input_dim = features_tensor.shape[1]
    autoencoder = SparseAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # DataLoader for batch processing
    dataset = TensorDataset(features_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Training the autoencoder
    num_epochs = 100  # May need adjustment
    best_loss = np.inf
    for epoch in range(num_epochs):
        for data in data_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(autoencoder.state_dict(), 'sparse_encoder.pt')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Encoding the features to reduce dimensionality
    autoencoder.load_state_dict(torch.load('sparse_encoder.pt'))
    reduced_features = autoencoder.encode(features_tensor).detach().numpy()
    
    return reduced_features

def compute_shortest_path_length(G, node_pairs):
    """
    Computes the shortest path length between pairs of nodes, ignoring direct edges.
    """
    shortest_paths = {}
    for source, target in node_pairs:
        try:
            # Temporarily remove the direct edge if it exists
            if G.has_edge(source, target):
                G.remove_edge(source, target)
                length = nx.shortest_path_length(G, source=source, target=target)
                G.add_edge(source, target)  # Add the edge back
            else:
                length = nx.shortest_path_length(G, source=source, target=target)
            shortest_paths[(source, target)] = length
        except nx.NetworkXNoPath:
            shortest_paths[(source, target)] = 1000  # No path exists
    return shortest_paths

def compute_number_of_paths(G, node_pairs, path_length=3):
    """
    Computes the number of paths of a given length between pairs of nodes.
    Note: This can be resource-intensive for large graphs or long path lengths.
    """
    num_paths = {}
    for source, target in node_pairs:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=path_length))
        num_paths[(source, target)] = len([p for p in paths if len(p)-1 == path_length])
    return num_paths

def rooted_pagerank(G, node, d = 0.85, epsilon = 1e-4):
    """ Returns rooted pagerank vector
    g graph
    node root
    d damping coefficient
    """
    ordered_nodes = sorted(G.nodes())
    root = ordered_nodes.index(node)
    adj = nx.to_numpy_array(G, nodelist = ordered_nodes)
    m = np.copy(adj)

    for i in range(len(G)):
        row_norm = np.linalg.norm(m[i], ord = 1)
        if row_norm != 0:
            m[i] = m[i] / row_norm

    m = m.transpose()

    rootvec = np.zeros(len(G))
    rootvec[root] = 1

    vect = np.random.rand(len(G))
    vect = vect / np.linalg.norm(vect, ord = 1)
    last_vect = np.ones(len(G)) * 100 # to ensure that does not hit epsilon randomly in first step

    iterations = 0
    while np.linalg.norm(vect - last_vect, ord = 2) > epsilon:
        last_vect = vect.copy()
        vect = d * np.matmul(m, vect) + (1 - d) * rootvec
        iterations += 1

    eigenvector = vect / np.linalg.norm(vect, ord = 1)

    eigen_dict = {}
    for i in range(len(ordered_nodes)):
        eigen_dict[ordered_nodes[i]] = eigenvector[i]

    return eigen_dict



def compute_save_rooted_pagerank_json(G, edge_list, damp, eps, trainval = False):
    # create dictionary to store result
    res = dict()

    # compute rooted pagerank
    
    pagerank = {root: rooted_pagerank(G, root, d=damp, epsilon=eps) for root in sorted(set([u for u, v in edge_list] + [v for u, v in edge_list]))}

    # Only store the edges we actually need in result dict
    for u, v in edge_list:
        res[str(u) + "_" + str(v)] = pagerank[u][v]

    # Save in json file
    if trainval:
        fname = f"rooted_pagerank_seed0_trainval_d{str(int(damp*100))}_eps{str(eps)}.json"
    else:
        fname = f"rooted_pagerank_seed0_test_d{str(int(damp*100))}_eps{str(eps)}.json"

    # Assuming you have a "data" directory. If not, adjust the path accordingly.
    with open(fname, "w") as file:
        json.dump(res, file)
        
def read_pagerank_json(pagerank_test, pagerank_trainval, u, v):
    key = str(u) + "_" + str(v)
    if key in pagerank_test.keys():
        return pagerank_test[key]
    elif key in pagerank_trainval.keys():
        return pagerank_trainval[key]



def hub_promoted_index(G, u, v):
    """
    Calculate the Hub Promoted Index (HPI) for a pair of nodes in a graph, handling cases where nodes have no neighbors.

    Parameters:
    - G: A NetworkX graph.
    - u, v: Nodes in the graph for which to calculate the HPI.

    Returns:
    - The HPI for the nodes u and v, or 0 if either node has no neighbors.
    """
    # Ensure u and v are in the graph
    if u not in G or v not in G:
        raise ValueError("Both nodes must be in the graph.")

    # Find the sets of neighbors for each node
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))

    # Check if either node has no neighbors
    if len(neighbors_u) == 0 or len(neighbors_v) == 0:
        return 0  # Return 0 if either node has no neighbors, as there can be no common neighbors

    # Calculate the intersection of the two neighbor sets
    intersection_size = len(neighbors_u & neighbors_v)

    # Calculate the HPI
    hpi = intersection_size / min(len(neighbors_u), len(neighbors_v))

    return hpi

def adamic_adar(G, source, target):
    # Check if source and target nodes exist in the graph
    if (source not in G.nodes() or target not in G.nodes()):
        return -1
    
    try:
        # Attempt the standard Adamic/Adar calculation
        preds = nx.adamic_adar_index(G, [(source, target)])
        for _, _, aa in preds:
            return aa
    except ZeroDivisionError:
        # Handle division by zero by excluding neighbors with degree of 1
        aa_score = 0
        for u, v in [(source, target)]:
            aa_score += sum(1 / np.log(G.degree(w)) for w in nx.common_neighbors(G, u, v) if G.degree(w) > 1)
        return aa_score
    
    
def salton_index_sparse(adj, G, u, v):
    
    

    # Ensure nodes are in the graph
    if u not in G or v not in G:
        return 0
    
    # Get the indices of u and v in the adjacency matrix
    u_idx, v_idx = list(G.nodes()).index(u), list(G.nodes()).index(v)
    
    # Compute the Salton Index (cosine similarity)
    u_vector, v_vector = adj[u_idx, :], adj[v_idx, :]
    
    numerator = u_vector.dot(v_vector.transpose()).toarray()[0][0]
    denominator = norm(u_vector) * norm(v_vector)
    
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def calculate_katz(G, node_pair, beta = 0.001, max_path_length = 3):
    paths = nx.all_simple_paths(G, source=node_pair[0], target=node_pair[1], cutoff=max_path_length)
    katz_score = 0
    for path in paths:
        katz_score += beta**len(path)
    return katz_score



def feature_extractor(fullgraph, train_graph, samples, node_feats):
    """    
    Creates a feature vector and a DataFrame for each edge in the samples.
    The DataFrame includes source_node, target_node, and computed features as columns,
    facilitating the creation of correlation graphs or further analysis.

    Parameters:
    - fullgraph: The complete graph from which global features are calculated.
    - train_graph: The graph used for training, from which local features are derived.
    - samples: Edges (or potential edges) for which features are calculated.
    - node_feats: Node features for additional metrics.
    
    Returns:
    - feature_vector: A list of numpy arrays, each representing an edge's features.
    - features_df: A DataFrame with columns for source_node, target_node, and other features.
    
    """
    feature_vector = []
    features_data = []
    
    with open('data/rooted_pagerank_seed0_test_d85_eps0.0001.json', 'r') as file:
        pagerank_test = json.load(file)

    with open('data/rooted_pagerank_seed0_trainval_d85_eps0.0001.json', 'r') as file:
        pagerank_trainval = json.load(file)
    # --- Graph-level Features ---
    # These features provide information about the entire graph.
    deg_centrality = nx.degree_centrality(fullgraph)  # Degree centrality for all nodes
    betweeness_centrality = nx.betweenness_centrality(fullgraph)  # Betweenness centrality for all nodes
    partition_louvain = community_louvain.best_partition(train_graph)  # Community detection
    
    # Convert undirected graph for specific metrics like triangles and clustering
    train_graph_ud = train_graph.to_undirected()
    triangles = nx.triangles(train_graph_ud)  # Number of triangles for each node
    clustering_coeff = nx.clustering(train_graph_ud)  # Clustering coefficient for each node
    
    # --- Precomputation for Edge-level Features ---
    # These precomputations support the calculation of edge-level features.
    nodelist = sorted(list(train_graph.nodes()))
    adj = nx.adjacency_matrix(train_graph, nodelist=nodelist)
    node_pairs = [(edge[0], edge[1]) for edge in samples]  # Edges in samples
    shortest_paths = compute_shortest_path_length(train_graph, node_pairs)
    num_paths_length_2 = compute_number_of_paths(train_graph, node_pairs, path_length=3)
    
    
    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]

        # --- Node-level Features ---
        # Features related to individual nodes' properties and their roles in the graph.
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]
        diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]
        source_triangles = triangles[source_node]
        target_triangles = triangles[target_node]
        source_coeff = clustering_coeff[source_node]
        target_coeff = clustering_coeff[target_node]

        # --- Edge-level Features ---
        # Features directly related to the relationship between two nodes.
        pref_attach = list(nx.preferential_attachment(train_graph, [(source_node, target_node)]))[0][2]
        louvain = 1 if partition_louvain[source_node] == partition_louvain[target_node] else 0
        aai = adamic_adar(train_graph, source_node, target_node)
        resource_alloc = list(nx.resource_allocation_index(train_graph, [(source_node, target_node)]))[0][2]
        cosine_sim = cosine_similarity(node_feats[int(source_node)].reshape(1, -1), node_feats[int(target_node)].reshape(1, -1))[0][0]
        jacc_coeff = list(nx.jaccard_coefficient(train_graph, [(source_node, target_node)]))[0][2]

        # --- Hybrid Features ---
        # Features that might involve multiple aspects of the graph, nodes, and edges.
        rooted_pagerank = read_pagerank_json(pagerank_test,  pagerank_trainval, source_node, target_node)
        SaI = salton_index_sparse(adj, train_graph, source_node, target_node)
        hub_prom_index = hub_promoted_index(train_graph, source_node, target_node)
        katz_index = calculate_katz(train_graph, (source_node, target_node))
        shortest_path_len = shortest_paths[(source_node, target_node)]  # May overfit
        num_paths = num_paths_length_2[(source_node, target_node)]  # May overfit
        
        # Compile feature vector for the current edge
        feature_vector.append(np.array([source_degree_centrality, target_degree_centrality,
                                        diff_bt, pref_attach, louvain, resource_alloc, jacc_coeff, 
                                        rooted_pagerank, SaI, aai, cosine_sim, katz_index, hub_prom_index, 
                                        shortest_path_len, num_paths, source_triangles, target_triangles, 
                                        source_coeff, target_coeff]))
        
        feature_row = {
            
            "source_node": source_node,
            "target_node": target_node,
            "source_degree_centrality": source_degree_centrality,
            "target_degree_centrality": target_degree_centrality,
            "diff_between_centrality": diff_bt,
            "preferential_attachment": pref_attach,
            "louvain_prop": louvain,
            "resource_alloc": resource_alloc,
            "jaccard_coeff": jacc_coeff,
            "rooted_page_rank": rooted_pagerank,
            "salton_index": SaI,
            "adamic_adar": aai,
            "cosine_sim": cosine_sim,
            "katz_index": katz_index,
            "hub_promoter_index": hub_prom_index,
            "shortest_path_len": shortest_path_len,
            "number_of_paths": num_paths,
            "source_triangles": source_triangles,
            "target_triangles": target_triangles,
            "source_clustering_coeff": source_coeff,
            "target_clustering_coeff": target_coeff
            
            
        }
        
        features_data.append(feature_row)
    
    features_df = pd.DataFrame(features_data)
        
    return feature_vector, features_df



def plot_lower_triangular_corr_matrix(features_df, labels):
    """
    Computes and plots the lower triangular correlation matrix from a given DataFrame.
    Drops 'source_node' and 'target_node' before computing the matrix.

    Parameters:
    - features_df: A DataFrame containing the features, including 'source_node' and 'target_node'.
    """
    # Drop the source and target node columns
    df_for_corr = features_df.drop(columns=['source_node', 'target_node'])
    df_for_corr['label'] = labels

    # Compute the correlation matrix
    corr = df_for_corr.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()

# Assuming `features_df` is your DataFrame with all the features including 'source_node' and 'target_node'
# plot_lower_triangular_corr_matrix(features_df)



def prediction(train_features_df, test_features_df, train_labels, test_labels):
    
    train_features = train_features_df.drop(columns=['source_node', 'target_node'])
    test_features = test_features_df.drop(columns=['source_node', 'target_node'])
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state=42)
    }
    
    plt.figure(figsize=(10, 10))
    results = {}
    colors = ['darkred', 'darkgreen', 'darkblue', 'darkorange']
    lw = 2
    
    # Train classifiers and compute ROC AUC
    for (i, (name, clf)) in enumerate(classifiers.items()):
        clf.fit(train_features, train_labels)
        test_preds = clf.predict_proba(test_features)[:, 1]
        
        fpr, tpr, _ = roc_curve(test_labels, test_preds)
        roc_auc = auc(fpr, tpr)
        results[name] = {'roc_auc': roc_auc, 'classifier': clf}
        
        plt.plot(fpr, tpr, color=colors[i], lw=lw, label=f'{name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")


    # Plot feature importance for RF and XGB
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for i, model_name in enumerate(['Random Forest', 'XGBoost']):
        model = results[model_name]['classifier']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax[i].set_title(f'Feature Importances ({model_name})')
        ax[i].bar(range(train_features.shape[1]), importances[indices], align='center')
        ax[i].set_xticks(range(train_features.shape[1]))
        ax[i].set_xticklabels(train_features.columns[indices], rotation=90)
        ax[i].set_xlim([-1, train_features.shape[1]])
    plt.tight_layout()
    plt.show()

    # Learning rate graph for RF and XGB (AUC by number of features)
    for model_name in ['Random Forest', 'XGBoost']:
        auc_scores = []
        model = results[model_name]['classifier']
        for n_features in range(1, train_features.shape[1] + 1):
            selected_features = train_features.columns[np.argsort(model.feature_importances_)[-n_features:]]
            clf = model.__class__(**model.get_params())
            clf.fit(train_features[selected_features], train_labels)
            test_preds = clf.predict_proba(test_features[selected_features])[:, 1]
            fpr, tpr, _ = roc_curve(test_labels, test_preds)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)

        plt.plot(range(1, train_features.shape[1] + 1), auc_scores, label=f'{model_name}')

    plt.xlabel('Number of Features')
    plt.ylabel('AUC')
    plt.title('AUC by Number of Features')
    plt.legend()
    plt.show()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    for i, (name, info) in enumerate(results.items()):
        ax = axes.flatten()[i]
        plot_confusion_matrix(info['classifier'], test_features, test_labels, ax=ax, cmap='Blues')
        ax.title.set_text(f"{name} (AUC: {info['roc_auc']:.4f})")
        ax.grid(False)
    plt.tight_layout()

    plt.show()
    
    return results



