import csv
import random
import numpy as np
import networkx as nx
from itertools import groupby
import itertools
import operator
import os
import numpy as np
import time
# importing random module
import random





import networkx as nx
from collections import deque
import random


class AttackModel:
    def __init__(self, target_fraction):
        """
        Initialize the SnowballSampler.

        :param graph: A NetworkX graph object representing the network.
        """
        self.target_fraction = target_fraction
        self.sampled_nodes = set()
        self.max_node_index = -1

    """
    def find_node_with_max_expansion(self, graph, seed, seed_list, neighbours_seedlist_with_seed):

        max_expanded_node = -1
        max_expansion = 0


        neighbours_seed =  list(graph.neighbors(seed))

        for node in neighbours_seed:
            neighbours_seedlist_with_seed[node] = True

        #p_lst = [node for node in range(self.max_node_index) if neighbours_seedlist_with_seed[node] == True]
        #print(len(p_lst),p_lst)


        for candidate_node in range(self.max_node_index-1,0,-1):
            if neighbours_seedlist_with_seed[candidate_node] == True and seed_list[candidate_node] == False:
                neighbours = list(graph.neighbors(candidate_node))

                max_criteria_len = 0
                for node in neighbours:
                    if neighbours_seedlist_with_seed[node] == False:
                        max_criteria_len = max_criteria_len + 1

                if max_criteria_len > max_expansion:
                    max_expansion = max_criteria_len
                    max_expanded_node = candidate_node

        if max_expanded_node != -1:
            seed_list[max_expanded_node] = True

        return max_expanded_node

    

    def snowball_sample(self, graph, seed_node):


        target_count = max(1, int(self.max_node_index * self.target_fraction))

        seed_list = np.zeros(self.max_node_index, dtype=bool)
        neighbours_seedlist_with_seed = np.zeros(self.max_node_index, dtype=bool)
        seed_list[seed_node] = True
        while np.sum(seed_list) < target_count:

            expanding_node = self.find_node_with_max_expansion(graph, seed_node, seed_list, neighbours_seedlist_with_seed)
            seed_node = expanding_node

            if expanding_node == -1:
                break

        seed_lst = [node for node in range(self.max_node_index) if seed_list[node] == True]
        #print(seed_lst)

        return seed_lst   

    """
    def find_node_with_max_expansion(self, graph, seed_list):
        max_expanded_node = -1
        max_expansion = 0
        neighbours_seedlist = set()

        for node in seed_list:
            neighbours_seedlist = neighbours_seedlist | set(list(graph.neighbors(node)))

        neighbours_seedlist = neighbours_seedlist - set(seed_list)

        neighbours_seedlist_with_seed = neighbours_seedlist | set(seed_list)

        #print(len(neighbours_seedlist_with_seed),neighbours_seedlist_with_seed)

        # expanding_node = seed_list[-1]
        # neighbours_expanding_node = list(graph.neighbors(expanding_node))

        for candidate_node in list(neighbours_seedlist):
            neighbours = list(graph.neighbors(candidate_node))

            max_criteria = list(set(neighbours) - set(neighbours_seedlist_with_seed))
            max_criteria_len = len(max_criteria)
            if max_criteria_len > max_expansion:
                max_expansion = max_criteria_len
                max_expanded_node = candidate_node

        if max_expanded_node != -1:
            seed_list.append(max_expanded_node)
        return max_expanded_node

    def snowball_sample(self, graph, seed_node):

        total_nodes = len(graph.nodes)
        target_count = max(1, int(total_nodes * self.target_fraction))

        seed_list = [seed_node]

        while seed_list and len(seed_list) < target_count:

            expanding_node = self.find_node_with_max_expansion(graph, seed_list)

            if expanding_node == -1:
                break


        return seed_list


    def edges_with_two_endpoint_in_nodes(self, graph, nodes):
        """
        Find edges in the graph where at least one endpoint is in the specified list of nodes.

        :param graph: A NetworkX graph object.
        :param nodes: A list or set of nodes.
        :return: A list of edges with at least one endpoint in the given nodes.
        """
        node_set = set(nodes)  # Convert to set for faster lookup
        edges = [
            edge for edge in graph.edges
            if edge[0] in node_set and edge[1] in node_set
        ]
        return edges
    def edges_with_one_endpoint_in_nodes(self, graph, nodes):
        """
        Find edges in the graph where at least one endpoint is in the specified list of nodes.

        :param graph: A NetworkX graph object.
        :param nodes: A list or set of nodes.
        :return: A list of edges with at least one endpoint in the given nodes.
        """
        node_set = set(nodes)  # Convert to set for faster lookup
        edges = set([
            edge for edge in graph.edges
            if edge[0] in node_set or edge[1] in node_set
        ]) - set([
            edge for edge in graph.edges
            if edge[0] in node_set and edge[1] in node_set
        ])

        return edges

    def subgraph_from_nodes(self, graph, nodes):
        edges_between = [(u, v) for u, v in graph.edges() if u in nodes and v in nodes]

        G = nx.Graph()

        # Add edges to the graph
        G.add_edges_from(edges_between)
        return G




    def attack_method(self, G, nodes):
        subgraph = self.subgraph_from_nodes(G, nodes)
        cloned_graph = subgraph.copy()

        core_numbers = nx.core_number(cloned_graph)

        # Find the maximum and second maximum core numbers
        sorted_core = sorted(set(core_numbers.values()), reverse=True)
        max_core = sorted_core[0]
        inner_core_nodes = [node for node, core in core_numbers.items() if core == max_core]

        #print(inner_core_nodes)
        second_inner_core_edges = set()
        intra_sec_inner_core_edges = set()
        second_inner_core_nodes = []
        if len(sorted_core) > 1:
            second_max_core = sorted_core[1]
            second_inner_core_nodes = [node for node, core in core_numbers.items() if core == second_max_core]
            second_inner_core_edges = set(self.edges_with_one_endpoint_in_nodes(G, second_inner_core_nodes))
            intra_sec_inner_core_edges = set(self.edges_with_two_endpoint_in_nodes(G, second_inner_core_nodes))

        # Get the subgraphs for the inner core and second inner core





        intra_inner_core_edges = set(self.edges_with_two_endpoint_in_nodes(G, inner_core_nodes))
        inner_core_edges = set(self.edges_with_one_endpoint_in_nodes(G, inner_core_nodes))



        # Compute intersection and exclusive edges
        intersection_edges = inner_core_edges & second_inner_core_edges
        remaining_inner_core_edges = inner_core_edges - intersection_edges - intra_inner_core_edges

        remaining_second_inner_core_edges = second_inner_core_edges - intersection_edges - intra_sec_inner_core_edges

        # Combine edges in the specified order
        edges_to_remove = (
                list(intra_inner_core_edges) +
                list(intersection_edges) +
                list(intra_sec_inner_core_edges) +
                list(remaining_inner_core_edges) +
                list(remaining_second_inner_core_edges)
        )

        return edges_to_remove, inner_core_nodes, second_inner_core_nodes
    """
    #prev implementation
    def find_node_with_max_expansion(self, graph, seed_list):

        max_expanded_node = -1
        max_expansion = 0
        neighbours_seedlist = set()
        for node in seed_list:
            neighbours_seedlist = neighbours_seedlist | set(list(graph.neighbors(node)))
        neighbours_seedlist_with_seed = neighbours_seedlist | set(seed_list)

        expanding_node = seed_list[-1]
        neighbours_expanding_node = list(graph.neighbors(expanding_node))

        for candidate_node in neighbours_expanding_node:
            neighbours = list(graph.neighbors(candidate_node))

            max_criteria = list(set(neighbours) - set(neighbours_seedlist_with_seed))
            max_criteria_len = len(max_criteria)
            if max_criteria_len > max_expansion:
                max_expansion = max_criteria_len
                max_expanded_node = candidate_node

        if max_expanded_node != -1:
            seed_list.append(max_expanded_node)
        return max_expanded_node

    """
