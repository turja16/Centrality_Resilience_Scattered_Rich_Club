import networkx as nx


from scatteredness import Scatteredness

scatteredness = Scatteredness()

files = scatteredness.get_file_names()

top_n = 20

for file in files:
    print("File:",file)
    G = scatteredness.build_graph_from_edgelist(file)
    #core_distribution = scatteredness.calculate_core_distribution(file, G, top_n)
    clusters = scatteredness.high_centrality_clusters(file, G, top_n)
    #print(len(clusters))
    deg_scatteredness = scatteredness.calculate_scatteredness(file, clusters)
    print(f"Degree of Scatteredness: {deg_scatteredness}")

    #snowball_core_distribution = scatteredness.snowball_core_distribution(file, G, top_n)


"""
# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)

    # Find clusters
    

    # Print results
    print(f"Number of clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {cluster}")




# Example usage
if __name__ == "__main__":
    # Example high-centrality nodes and clusters
    high_centrality_nodes = {1, 2, 3, 4, 5}  # Example Nhc
    clusters = [
        {1, 6, 7},  # Cluster 1
        {2, 3, 8},  # Cluster 2
        {4, 9, 10},  # Cluster 3
        {5, 11}  # Cluster 4
    ]

    # Calculate degree of scatteredness
    scatteredness = calculate_scatteredness(high_centrality_nodes, clusters)
    print(f"Degree of Scatteredness: {scatteredness}")
"""
