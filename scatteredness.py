"""
Simulation.
"""
import networkx as nx
import sys as sys
import os

import random
import time
from copy import deepcopy
from metrics import Jaccard_Index
from snowball import AttackModel
import math
import csv
from simulate import AttackSimulator
#import matplotlib.pyplot as plt
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
from collections import Counter

class Scatteredness:
    """
    TODO Till now, only directed graph is supported.
    TODO Till now, only nodes/edges removal is supported. Perturbation is coming soon.
    """

    def __init__(
            self
    ) -> None:
        """
        Args:
            how (str):
        """
        self.input_folder = "Datasets"
        self.centrality_folder = "Datasets"
        self.output_folder = "output"
        self.centrality_folder = "centrality"
        self.hdeg_hcoeff_nodes = []
        self.base_top_k_bet_cen = []
        self.base_top_k_close_cen = []
        self.max_node_index = -1

        self.write_header()
    """
    def build_graph_from_edgelist(self, filename):
    
        # Create an empty graph
        G = nx.Graph()
        filename = filename + ".txt"
        filepath = os.path.join(self.input_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, 'r') as file:
            for line in file:
                # Each line should contain two nodes separated by whitespace (e.g., "node1 node2")
                nodes = line.strip().split()
                if len(nodes) == 2:  # Ensure it has exactly two elements
                    G.add_edge(nodes[0], nodes[1])
        G.remove_edges_from(nx.selfloop_edges(G))



        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G
    """

    def build_graph_from_edgelist(self, filename):
        """Read an edgelist from a file and create a NetworkX graph."""
        # Create an empty graph
        G = nx.Graph()
        filename = filename + ".txt"
        filepath = os.path.join(self.input_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, 'r') as file:
            for line in file:
                # Each line should contain two nodes separated by whitespace (e.g., "node1 node2")
                nodes = line.strip().split()
                if len(nodes) == 2:  # Ensure it has exactly two elements
                    start, end = int(nodes[0]), int(nodes[1])
                    G.add_edge(start, end)

                    if start > self.max_node_index:
                        self.max_node_index = start
                    if end > self.max_node_index:
                        self.max_node_index = end

        self.max_node_index = self.max_node_index  + 1
        G.remove_edges_from(nx.selfloop_edges(G))

        #print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G









    def calculate_node_score(self, graph, node, clustering_coefficients, max_degree, max_clustering):
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
        # score = normalized_degree * normalized_clustering
        # score = 2 * (.6 *normalized_degree * .4* normalized_clustering) / (.6 *normalized_degree + .4* normalized_clustering)

        return score

    def get_top_nodes_by_score(self, graph, top_n=40):
        """
        Calculate scores for all nodes, sort them, and return the top N nodes.
        """
        # Calculate scores for all nodes
        max_degree = max(dict(graph.degree()).values())
        clustering_coefficients = nx.clustering(graph)
        max_clustering = max(clustering_coefficients.values())
        node_scores = {node: self.calculate_node_score(graph, node, clustering_coefficients, max_degree, max_clustering)
                       for node in graph.nodes()}

        # Sort nodes by score in descending order
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        # Get the top N nodes
        top_nodes = sorted_nodes[:top_n]
        # print(top_nodes)
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



    def write_header(self):
        """Write the header to a CSV file in a specified folder."""
        fieldnames = ["Name", "Distribution", "Scatteredness"]
        filename = "scatteredness" + ".txt"
        self.scfilepath = os.path.join(self.output_folder, filename)  # Full path to file
        with open(self.scfilepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

        filename = "core_distribution" + ".txt"
        fieldnames = ["Name", "Core_Distribution"]
        self.corefilepath = os.path.join(self.output_folder, filename)  # Full path to file
        with open(self.corefilepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

        filename = "snowball_core_distribution" + ".txt"
        fieldnames = ["Name", "Core_Distribution"]
        self.snowfilepath = os.path.join(self.output_folder, filename)  # Full path to file
        with open(self.corefilepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()


    def write_values(self, filepath,data):
        """Append rows of data to a CSV file in a specified folder."""

        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writerows(data)




    def get_file_names(self):
        """Rename all .txt files in a folder by removing the .txt extension."""
        files = []
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.txt'):
                new_filename = filename[:-4]  # Remove the last 4 characters (".txt")
                files.append(new_filename)
        return files

    def load_top_betweenness_centrality_nodes(self, file, top_n=20):
        filename = "bet_centrality_"+file + ".txt"
        filepath = os.path.join(self.centrality_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, "r") as file:
            lines = file.readlines()  # Read all lines
        top_nodes = [line.split("\t")[0] for line in lines[1:top_n+1] if line.strip()]  # Skip header and process non-empty lines

        #print(top_nodes)
        return top_nodes
    def top_betweenness_centrality_nodes(self, G, top_n=20):
        """Find the top N nodes with the highest betweenness centrality in the graph."""
        # Calculate betweenness centrality for all nodes
        centrality = nx.betweenness_centrality(G)

        # Sort nodes by centrality in descending order and get the top N
        top_cent_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Print the top N nodes and their betweenness centrality values
        top_nodes = []
        for node, centrality_value in top_cent_nodes:
            #print(f"Node: {node}, Betweenness Centrality: {centrality_value}")
            top_nodes.append(node)

        output_file = "betweenness_centrality.txt"
        with open(output_file, "a") as file:
            file.write("Node\tBetweenness Centrality\n")
            for node, centrality in top_cent_nodes:
                file.write(f"{node}\t{centrality:.6f}\n")

        return top_nodes

    def load_top_closeness_centrality_nodes(self, file, top_n=20):
        filename = "close_centrality_"+file + ".txt"
        filepath = os.path.join(self.centrality_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, "r") as file:
            lines = file.readlines()  # Read all lines
        top_nodes = [line.split("\t")[0] for line in lines[1:top_n+1] if line.strip()]  # Skip header and process non-empty lines

        #print(top_nodes)
        return top_nodes

    def top_closeness_centrality_nodes(self, G, top_n=20):
        """Find the top N nodes with the highest closeness centrality in the graph."""
        # Calculate closeness centrality for all nodes
        centrality = nx.closeness_centrality(G)


        # Sort nodes by centrality in descending order and get the top N
        top_cent_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True) [:top_n]

        # Print the top N nodes and their betweenness centrality values
        top_nodes = []
        for node, centrality_value in top_cent_nodes:
            #print(f"Node: {node}, closeness_centrality: {centrality_value}")
            top_nodes.append(node)
            #print(f"Node: {node}, degree: {G.degree(node) }")

        output_file = "closeness_centrality.txt"
        with open(output_file, "a") as file:
            file.write("Node\tcloseness Centrality\n")
            for node, centrality in top_cent_nodes:
                file.write(f"{node}\t{centrality:.6f}\n")

        return top_nodes

    def overlapping_neighbours_ratio(self, first_clus_seed, other_seed, graph ):

        first_neighbours = []
        other_neighbours = []


        for seed in list(first_clus_seed):
            first_neighbours = first_neighbours + list(graph.neighbors(seed))

        for seed in list(other_seed):
            other_neighbours = other_neighbours + list(graph.neighbors(seed))

        first_neighbours_set = set(first_neighbours) - set(first_clus_seed) - set(other_seed)
        other_neighbours_set = set(other_neighbours) - set(first_clus_seed) - set(other_seed)

        interset_set = (first_neighbours_set & other_neighbours_set)

        """
        ref_first_clus_seed = []

        for to in interset_set:
            for frm in first_neighbours_set:
                if (to,frm) in graph.edges:
                    ref_first_clus_seed.append(frm)
        ref_other_seed = []
        for to in interset_set:
            for frm in other_neighbours_set:
                if (to,frm) in graph.edges:
                    ref_other_seed.append(frm)

        ref_first_neighbours = []
        ref_other_neighbours = []


        for seed in list(ref_first_clus_seed):
            ref_first_neighbours = first_neighbours + list(graph.neighbors(seed))

        for seed in list(ref_other_seed):
            ref_other_neighbours = other_neighbours + list(graph.neighbors(seed))

        first_neighbours_set = set(ref_first_neighbours) - set(first_clus_seed) - set(other_seed)
        other_neighbours_set = set(ref_other_neighbours) - set(first_clus_seed) - set(other_seed)
        """

        union_set_len = len(first_neighbours_set | other_neighbours_set)
        interset_set_len = len(first_neighbours_set & other_neighbours_set)

        if union_set_len != 0:
            ratio = interset_set_len/union_set_len
        else:
            ratio = 0
        return ratio



    def merge_overlapping_sets_recursive(self, id2cluster, id2hcn, graph):
        """
        Recursively merges overlapping clusters in id2cluster and updates id2hcn synchronously.

        Parameters:
            id2cluster (dict): A dictionary mapping IDs to clusters (sets of nodes).
            id2hcn (dict): A dictionary mapping IDs to a set of nodes.

        Returns:
            tuple: Updated id2cluster and id2hcn where overlapping clusters are merged.
        """
        if not id2cluster:  # Base case: if id2cluster is empty, return both as is
            return id2cluster, id2hcn

        # Get the first cluster (by its ID) and the rest of the clusters
        first_id = next(iter(id2cluster))
        first_set = id2cluster[first_id]
        rest_clusters = {k: v for k, v in id2cluster.items() if k != first_id}

        # Find overlapping clusters
        overlapping = {}
        for k, v in rest_clusters.items():
            other_seed = id2hcn[k]
            if not first_set.isdisjoint(other_seed):
                first_seed = id2hcn[first_id]
                ratio = self.overlapping_neighbours_ratio(first_seed,other_seed,graph)
                if ratio > .1:
                    overlapping[k] = v

        # Handle the non-overlapping clusters separately
        non_overlapping_id2cluster = {}
        non_overlapping_id2hcn = {}
        for k, v in rest_clusters.items():
            if k not in overlapping:
                non_overlapping_id2cluster[k] = v
                non_overlapping_id2hcn[k] = id2hcn[k]

        # Merge overlapping clusters into the first cluster
        for overlap_id, overlap_set in overlapping.items():
            first_set = first_set.union(overlap_set)
            # Merge id2hcn entries
            id2hcn[first_id] = id2hcn[first_id] | id2hcn[overlap_id]
            id2cluster[first_id] = id2cluster[first_id] | id2cluster[overlap_id]
            # Remove the merged cluster from id2hcn and id2cluster
            del id2hcn[overlap_id]
            del id2cluster[overlap_id]




        # Recur on the remaining non-overlapping clusters
        remaining_id2cluster, remaining_id2hcn = self.merge_overlapping_sets_recursive(non_overlapping_id2cluster, non_overlapping_id2hcn, graph)

        remaining_id2cluster[first_id] = first_set
        remaining_id2hcn[first_id] = id2hcn[first_id]



        return remaining_id2cluster, remaining_id2hcn

    def check_overlaps(self, G, sets):
        """
        Checks if any sets in the list overlap.

        Parameters:
            sets (list of sets): A list of sets to check.

        Returns:
            bool: True if any sets overlap, False otherwise.
        """
        for st in sets:
            for item in st:
                neighbours = list(G.neighbors(item))
                neighbours_set = set(neighbours)
                remaining_set = st - {item}
                if len(remaining_set) ==0:
                    continue
                if neighbours_set.isdisjoint(remaining_set) and item not in remaining_set:  # Check for overlap
                    print('no overlap for ',item)
                if len(neighbours_set & remaining_set) < 1:
                    sys.exit(1)

    def calculate_core_distribution(self, file, G, top_n=10):
        """
        Calculate the core distribution of high-centrality nodes in a graph.
        Args:
            file (str): File name for reading top centrality nodes.
            G (networkx.Graph): Input graph.
            top_n (int): Number of top centrality nodes to consider.

        Returns:
            dict: Core distribution in percentages, sorted from highest to lowest core number.
        """
        # Load top centrality nodes
        top_betweenness = self.load_top_betweenness_centrality_nodes(file, top_n)
        top_closeness = self.load_top_closeness_centrality_nodes(file, top_n)

        # Unified set of high-centrality vertices
        Nhc = set(top_betweenness).union(top_closeness)

        Nhc = [int(node) for node in Nhc]

        # Calculate core numbers
        core_numbers = nx.core_number(G)

        max_core_number = max(core_numbers.values())

        # Print the maximum core number
        print("File:", file)
        print("Maximum Core Number:", max_core_number)

        # Compute raw core distribution
        core_distribution = {}
        for node in Nhc:
            core = core_numbers[node]
            core_distribution[core] = core_distribution.get(core, 0) + 1

        print("Core Distribution:", core_distribution)
        data = [
            {"Name": file, "Core_Distribution": core_distribution},
        ]
        self.write_values(self.corefilepath, data)

        # Calculate percentages
        total_nodes = sum(core_distribution.values())
        core_distribution_percentage = {
            core: (count / total_nodes) * 100
            for core, count in core_distribution.items()
        }

        # Sort the distribution by core number in descending order
        sorted_core_distribution = dict(sorted(
            core_distribution_percentage.items(),
            key=lambda item: item[0],
            reverse=True
        ))

        # Print and save the results
        print("Sorted Core Distribution (Percentage):", sorted_core_distribution)
        data = [
            {"Name": file, "Core_Distribution": sorted_core_distribution},
        ]
        self.write_values(self.corefilepath, data)

        return sorted_core_distribution


    def snowball_core_distribution(self, file, G, top_n=10):
        """
        Calculate the core distribution of high-centrality nodes in a graph.
        Args:
            file (str): File name for reading top centrality nodes.
            G (networkx.Graph): Input graph.
            top_n (int): Number of top centrality nodes to consider.

        Returns:
            dict: Core distribution in percentages, sorted from highest to lowest core number.
        """
        # Load top centrality nodes
        top_betweenness = self.load_top_betweenness_centrality_nodes(file, top_n)
        top_closeness = self.load_top_closeness_centrality_nodes(file, top_n)

        # Unified set of high-centrality vertices
        Nhc = set(top_betweenness).union(top_closeness)
        Nhc = [int(node) for node in Nhc]

        # Calculate core numbers
        jac_Index = Jaccard_Index()
        attack_model = AttackSimulator(jac_Index, file)
        attack_model.attackmodel.max_node_index = self.max_node_index
        seed = attack_model.find_high_degree_high_clustering_nodes(G)
        nodes = attack_model.attackmodel.snowball_sample(G,seed)
        snow_graph = G.subgraph(nodes)
        core_numbers = nx.core_number(snow_graph)

        max_core_number = max(core_numbers.values())

        # Print the maximum core number
        print("File:", file)
        print("Maximum Core Number:", max_core_number)

        # Compute raw core distribution
        core_distribution = {}
        for node in Nhc:
            if node in nodes:
                core = core_numbers[node]
                core_distribution[core] = core_distribution.get(core, 0) + 1


        print("Core Distribution:", core_distribution)
        data = [
            {"Name": file, "Core_Distribution": core_distribution},
        ]
        self.write_values(self.snowfilepath, data)

        # Calculate percentages
        #total_nodes = sum(core_distribution.values())
        total_nodes = len(Nhc)
        core_distribution_percentage = {
            core: (count / total_nodes) * 100
            for core, count in core_distribution.items()
        }

        # Sort the distribution by core number in descending order
        sorted_core_distribution = dict(sorted(
            core_distribution_percentage.items(),
            key=lambda item: item[0],
            reverse=True
        ))

        # Print and save the results
        print("Sorted Core Distribution (Percentage):", sorted_core_distribution)
        data = [
            {"Name": file, "Core_Distribution": sorted_core_distribution},
        ]
        self.write_values(self.snowfilepath, data)

        return sorted_core_distribution




    def high_centrality_clusters(self, file, G, top_n=10):

        top_betweenness = self.load_top_betweenness_centrality_nodes(file, top_n)
        top_closeness = self.load_top_closeness_centrality_nodes(file, top_n)

        # Unified set of high-centrality vertices
        Nhc = set(top_betweenness).union(top_closeness)
        Nhc = [int(node) for node in Nhc]
        id2hcn = {}
        id2cluster = {}

        for idx, v in enumerate(Nhc):
            cluster = set(G.neighbors(v))
            id2cluster[idx] = cluster
            id2hcn[idx] = set([v])

        prev = len(id2hcn)
        curr = 0
        while prev != curr:
            prev = len(id2hcn)
            id2cluster, id2hcn= self.merge_overlapping_sets_recursive(id2cluster, id2hcn, G)
            curr = len(id2hcn)


        clusters = [cluster for id, cluster in id2hcn.items()]
        self.check_overlaps(G,clusters)


        print("Non-overlapping sets:", clusters)

        return clusters


    def calculate_scatteredness(self,file, clusters):
        """
        Calculate the degree of scatteredness.

        Parameters:
        - high_centrality_nodes: Set of high-centrality nodes (Nhc).
        - clusters: List of clusters (list of sets) from the previous step.

        Returns:
        - Degree of scatteredness (float).
        """
        # Total number of high-centrality nodes (H)
        size_clusters = [len(cluster) for cluster in clusters]
        frequency_clusters = Counter(size_clusters)
        H = sum([len(cluster) for cluster in clusters])
        K = len(clusters)  # Total number of clusters
        R = []
        """
        for key, value in frequency_clusters.items():
            Ri = key/(H*value)
            R.append(Ri)






        """
        # Count high-centrality nodes in each cluster
        high_centrality_counts = [
            len(cluster) for cluster in clusters
        ]
        print(f'Distribution: {high_centrality_counts}')

        item_counts = Counter(high_centrality_counts)

        output_parts = []
        for item, count in item_counts.items():
            output_parts.append(f"{count}({item})")

        output_string = ", ".join(output_parts)
        output_string = output_string + " " + str(high_centrality_counts)
        print(output_string)


        # Sort counts in descending order (Hi ≥ Hj for i < j)
        high_centrality_counts.sort(reverse=True)

        # Calculate Ri for each cluster
        seen_high_centrality_nodes = 0
        unseen_high_centrality_nodes = H
        seen_counter = 1
        #Z = 0
        
        for Hx in high_centrality_counts:
            if H == Hx:
                Ri = 1
                R.append(Ri)
            elif Hx > 0:

                Ri = Hx /(H*seen_counter)

                R.append(Ri)
                seen_high_centrality_nodes += Hx
                unseen_high_centrality_nodes -= Hx
                seen_counter = seen_counter + 1
        


        mean = sum(R)

        self.fieldnames = ["Name", "Distribution", "Scatteredness"]
        data = [
            {"Name": file, "Distribution": output_string, "Scatteredness": mean},
        ]
        self.write_values(self.scfilepath, data)
        return mean
