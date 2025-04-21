"""
Simulation.
"""
import networkx as nx
import os
import random
import time
from copy import deepcopy
from metrics import Jaccard_Index
from snowball import AttackModel
import math

from typing import Union, List, Optional, Callable
from utils import (
    datadict,
    tolist,
    list_mean,
    list_dim,
    remove_random_egdes,
    remove_random_nodes,
    remove_targeted_edges,
    remove_targeted_nodes
)


class AttackSimulator:
    """
    TODO Till now, only directed graph is supported.
    TODO Till now, only nodes/edges removal is supported. Perturbation is coming soon.
    """
    def __init__(
        self,
        jacIndex:Jaccard_Index,
        filename: str
    ) -> None:
        """
        Args:
            how (str):
        """
        self.jacIndex = jacIndex
        self.methods = ['ours_hdhc']  #, 'core_decomposition','degree_product', 'betweenness' , 'core_decomposition', 'betweenness'
        self.filename = filename
        self.input_folder = "Datasets"
        self.attackmodel = AttackModel(.1)
        self.hdeg_hcoeff_nodes =  []
        self.max_node_index = -1





    def build_graph_from_edgelist(self, top_n):
        """Read an edgelist from a file and create a NetworkX graph."""
        # Create an empty graph
        self.G = nx.Graph()
        filename = self.filename + ".txt"
        filepath = os.path.join(self.input_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, 'r') as file:
            for line in file:
                # Each line should contain two nodes separated by whitespace (e.g., "node1 node2")
                nodes = line.strip().split()
                if len(nodes) == 2:  # Ensure it has exactly two elements
                    start, end = int(nodes[0]), int(nodes[1])
                    self.G.add_edge(start, end)

                    if start > self.max_node_index:
                        self.max_node_index = start
                    if end > self.max_node_index:
                        self.max_node_index = end

        self.max_node_index = self.max_node_index  + 1
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        #self.jacIndex.set_base_cent_centrality_nodes(self.G)
        self.jacIndex.load_base_cent_centrality_nodes(self.filename, top_n)
        self.attackmodel.max_node_index = self.max_node_index

        print(f"Graph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        filename = "network_info"
        filepath = os.path.join(self.input_folder, filename)  # Full path to file
        with open(filepath, mode='a', newline='') as file:
            file.write(f"{self.filename}: Graph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    def calculate_node_score(self, graph, node, clustering_coefficients,max_degree,max_clustering ):
        """
        Calculate a score for a node based on its normalized degree and normalized clustering coefficient.
        """
        # Normalized degree
        degree = graph.degree(node)

        normalized_degree = degree / max_degree if max_degree > 0 else 0

        # Normalized clustering coefficient


        normalized_clustering = clustering_coefficients[node] / max_clustering if max_clustering > 0 else 0

        # Final score
        score = (.7 * normalized_degree + .3 * normalized_clustering)
        #score = normalized_degree * normalized_clustering
        #score = 2 * (.6 *normalized_degree * .4* normalized_clustering) / (.6 *normalized_degree + .4* normalized_clustering)

        return score

    def get_top_nodes_by_score(self,graph, top_n=40):
        """
        Calculate scores for all nodes, sort them, and return the top N nodes.
        """
        # Calculate scores for all nodes
        max_degree = max(dict(graph.degree()).values())
        clustering_coefficients = nx.clustering(graph)
        max_clustering = max(clustering_coefficients.values())
        node_scores = {node: self.calculate_node_score(graph, node, clustering_coefficients,max_degree,max_clustering ) for node in graph.nodes()}

        # Sort nodes by score in descending order
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        # Get the top N nodes
        top_nodes = sorted_nodes[:top_n]
        #print(top_nodes)
        top_nodes = [tup[0] for tup in top_nodes]
        self.hdeg_hcoeff_nodes = top_nodes

    def find_high_degree_high_clustering_nodes(self, graph):
        """
        Find nodes with high degree and high clustering coefficient.

        :param graph: A NetworkX graph object.
        """
        self.get_top_nodes_by_score(graph)
        seed = self.hdeg_hcoeff_nodes[0]


        return seed




    def calculate_edges_todrop(self, G, method, drop_num):
        """
        Calculate edge priorities based on degree product or edge betweenness centrality.

        Parameters:
        - G (networkx.Graph): The input graph.
        - method (str): The method to calculate priorities. Options are:
            - 'degree_product': Priorities based on the product of the degrees of each edge's endpoints.
            - 'betweenness': Priorities based on edge betweenness centrality.

        Returns:
        - dict: A dictionary with edges as keys and their priority values as values.
        """
        edge_priorities = {}
        print(method)

        if method == 'degree_product':
            # Calculate priority based on degree product for each edge
            for u, v in G.edges():
                priority = G.degree[u] * G.degree[v]
                edge_priorities[(u, v)] = priority

            drop_edges = dict(sorted(edge_priorities.items(), key=lambda item: item[1], reverse=True))

            drop_edges = [edge for edge in drop_edges][:drop_num]
        
        elif method == 'betweenness':
            # Calculate edge betweenness centrality for each edge
            edge_betweenness = nx.edge_betweenness_centrality(G)
            for edge, centrality in edge_betweenness.items():
                edge_priorities[edge] = centrality

            drop_edges = dict(sorted(edge_priorities.items(), key=lambda item: item[1], reverse=True))

            drop_edges = [edge for edge in drop_edges][:drop_num]

        elif method == 'core_decomposition':
            # Calculate priority based on summation of corenumber for each edge
            core_number = nx.core_number(G)
            for u, v in G.edges():
                priority = max(core_number[u], core_number[v])
                edge_priorities[(u, v)] = priority

            drop_edges = dict(sorted(edge_priorities.items(), key=lambda item: item[1], reverse=True))

            drop_edges = [edge for edge in drop_edges][:drop_num]

        elif method == 'ours_random':
            # Calculate priority based on summation of corenumber for each edge
            seed = random.choice(list(G.nodes))
            print(f'Seed node:{seed}')
            nodes = self.attackmodel.snowball_sample(G, seed)
            edges_to_remove, inner_core_nodes = self.attackmodel.attack_method(G, nodes)
            drop_edges = [edge for edge in edges_to_remove][:drop_num]
            return drop_edges, inner_core_nodes
        elif method == 'ours_hdhc':
            # Calculate priority based on summation of corenumber for each edge
            seed = self.find_high_degree_high_clustering_nodes(G)
            #print(f'Seed node:{seed}')

            nodes = self.attackmodel.snowball_sample(G, seed)
            edges_to_remove,inner_core_nodes, second_inner_core_nodes = self.attackmodel.attack_method(G, nodes)
            drop_edges = [edge for edge in edges_to_remove][:drop_num]
            return drop_edges


        else:
            raise ValueError("Method must be 'degree_product' or 'betweenness'.")

        return drop_edges




    def step(self, graph: nx.Graph, method: str, drop_num: int):
        """
        Attack for single graph.
        """
        g = deepcopy(graph)
        drop_edges = self.calculate_edges_todrop(g, method, drop_num)

        return drop_edges



    def attack( self):
        """
        Args:
            filename (str): number of nodes/edges are attacked

        """

        total_edges = self.G.number_of_edges()
        percentages = [2,4,6,8]


        for method in self.methods:
            for percent in percentages:
                attacked_net = self.G.copy()
                edge_counts = math.ceil(total_edges * (percent / 100))
                start = time.time()
                drop_edges = self.step(attacked_net, method, edge_counts)  # attack the graph
                remove_targeted_edges(attacked_net, drop_edges)
                end = time.time()
                diff = end - start

                timedata = [
                    {"Method": method, "edge removal percentage": percent, "jaccard index": diff},
                ]
                self.jacIndex.write_values(self.filename,"time", timedata)
                #print(f'percent {percent}  Number of edges:{attacked_net.number_of_edges()}')
                self.jacIndex.calculate_jaccard_index(attacked_net, self.filename, method, percent)


    def predict(self, top_n):
        self.get_top_nodes_by_score(self.G,top_n)
        # print(f'Seed node:{seed}')
        predicted_high_cent_nodes = set()
        for seed in self.hdeg_hcoeff_nodes:
            self.attackmodel.max_node_index = self.max_node_index
            nodes = self.attackmodel.snowball_sample(self.G, seed)
            _, inner_core_nodes, second_inner_core_nodes = self.attackmodel.attack_method(self.G, nodes)
            predicted_high_cent_nodes = predicted_high_cent_nodes|set(inner_core_nodes)|set(second_inner_core_nodes)

        return predicted_high_cent_nodes




    def calculate_precision_recall(self, top_n):
        actual_high_cent_nodes =  set(self.jacIndex.base_top_k_bet_cen)|set(self.jacIndex.base_top_k_close_cen)
        predicted_high_cent_nodes = self.predict(top_n)


        # All nodes in the union of both sets
        all_nodes = actual_high_cent_nodes.union(predicted_high_cent_nodes)

        # Calculate True Positives (TP)
        tp = len(actual_high_cent_nodes.intersection(predicted_high_cent_nodes))

        # Calculate False Positives (FP)
        fp = len(predicted_high_cent_nodes - actual_high_cent_nodes)

        # Calculate False Negatives (FN)
        fn = len(actual_high_cent_nodes - predicted_high_cent_nodes)

        # Calculate True Negatives (TN)
        #tn = len(all_nodes - actual_high_cent_nodes - predicted_high_cent_nodes)
        tn = len(all_nodes) - (tp + fp + fn)

        # Calculate Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Calculate Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate Accuracy
        accuracy = (tp + tn) / len(all_nodes) if len(all_nodes) > 0 else 0.0

        # Print the results
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

        return precision, recall

