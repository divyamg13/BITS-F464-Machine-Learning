import pandas as pd
import networkx as nx
import numpy as np

def generate_ex_graph_features(input_bq_csv, output_edges_csv, output_nodes_csv):
    print(f"Loading data from {input_bq_csv}...")
    df = pd.read_csv(input_bq_csv)
    
    # 1. Aggregate transactions to get edge weights (number of transactions between a pair)
    edge_counts = df.groupby(['from_address', 'to_address']).size().reset_index(name='weight')
    
    # 2. Create a directed graph from the transactions
    print("Building transaction graph...")
    G = nx.from_pandas_edgelist(
        edge_counts, 
        source='from_address', 
        target='to_address', 
        edge_attr='weight', 
        create_using=nx.DiGraph()
    )
    
    # Create an undirected version to easily calculate overall unique neighbors
    G_undirected = nx.Graph(G)
    
    print("Calculating the 8 structural features...")
    # Feature 1 & 2: In-degree and Out-degree (total incoming/outgoing transactions)
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    
    # Feature 0: Degree (total transactions sent and received)
    degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in G.nodes()}
    
    # Feature 4 & 5: In-neighbors and Out-neighbors (unique sender/receiver addresses)
    in_neighbors = {n: len(list(G.predecessors(n))) for n in G.nodes()}
    out_neighbors = {n: len(list(G.successors(n))) for n in G.nodes()}
    
    # Feature 3: Neighbors (total unique connected addresses)
    neighbors = {n: len(list(G_undirected.neighbors(n))) for n in G.nodes()}
    
    # Feature 6: Max transactions with neighbors
    max_txns = {}
    for n in G.nodes():
        # Get transaction counts (weights) for all incoming and outgoing edges
        in_weights = [d['weight'] for _, _, d in G.in_edges(n, data=True)]
        out_weights = [d['weight'] for _, _, d in G.out_edges(n, data=True)]
        all_weights = in_weights + out_weights
        max_txns[n] = max(all_weights) if all_weights else 0
        
    # Feature 7: Average neighbor degree
    avg_neighbor_degree = {}
    for n in G.nodes():
        nbrs = list(G_undirected.neighbors(n))
        if len(nbrs) > 0:
            avg_neighbor_degree[n] = np.mean([degree[nbr] for nbr in nbrs])
        else:
            avg_neighbor_degree[n] = 0.0

    # --- 3. Mapping to Node IDs ---
    print("Mapping addresses to node IDs...")
    unique_addresses = list(G.nodes())
    address_to_id = {addr: idx for idx, addr in enumerate(unique_addresses)}
    
    # --- 4. Create Nodes DataFrame ---
    print(f"Creating nodes file: {output_nodes_csv}...")
    nodes_data = []
    for addr in unique_addresses:
        nodes_data.append({
            'node_id': address_to_id[addr],
            'label': 0, # Background/unclassified nodes default to 0
            'features_0': degree[addr],
            'features_1': in_degree.get(addr, 0),
            'features_2': out_degree.get(addr, 0),
            'features_3': neighbors[addr],
            'features_4': in_neighbors[addr],
            'features_5': out_neighbors[addr],
            'features_6': max_txns[addr],
            'features_7': avg_neighbor_degree[addr]
        })
        
    nodes_df = pd.DataFrame(nodes_data)
    
    # Ensure correct column order
    expected_cols = [
        'node_id', 'label', 
        'features_0', 'features_1', 'features_2', 'features_3', 
        'features_4', 'features_5', 'features_6', 'features_7'
    ]
    nodes_df = nodes_df[expected_cols]
    nodes_df.to_csv(output_nodes_csv, index=False)
    
    # --- 5. Create Edges DataFrame ---
    print(f"Creating edges file: {output_edges_csv}...")
    # We use the raw original dataframe to preserve all individual transaction edges
    df['src_node_id'] = df['from_address'].map(address_to_id)
    df['dst_node_id'] = df['to_address'].map(address_to_id)
    edges_df = df[['src_node_id', 'dst_node_id']]
    edges_df.to_csv(output_edges_csv, index=False)
    
    print(f"Success! Processed {len(nodes_df)} nodes and {len(edges_df)} edges.")

# --- Execution ---
if __name__ == "__main__":
    # You may need to pip install networkx if you haven't already
    INPUT_BQ = 'bq-results-20260421-170236-1776792206624.csv'
    OUTPUT_EDGES = 'ex_graph_edges.csv'
    OUTPUT_NODES = 'ex_graph_nodes.csv'
    
    generate_ex_graph_features(INPUT_BQ, OUTPUT_EDGES, OUTPUT_NODES)