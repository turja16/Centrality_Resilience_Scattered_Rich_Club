"""
Supplementary metrics.
"""
import networkx as nx
import csv
import os
import math

class Jaccard_Index:
    def __init__(
        self
    ) -> None:
        self.base_top_k_bet_cen = []
        self.base_top_k_close_cen = []
        self.fieldnames = ["Method", "edge removal percentage", "jaccard index"]
        self.input_folder = "Datasets"
        self.output_folder = "output"
        self.centralities = ["betweenness", "closeness"]
        self.write_header()
        self.centrality_folder = "centrality"

    def write_header(self):
        """Write the header to a CSV file in a specified folder."""
        os.makedirs(self.output_folder, exist_ok=True)  # Create folder if it doesn't exist
        files = self.get_file_names()
        for file_name in files:
            for centrality in self.centralities:
                filename = centrality + "_" + file_name + ".csv"
                filepath = os.path.join(self.output_folder, filename)  # Full path to file
                with open(filepath, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                    writer.writeheader()


    def write_values(self, filename, prefix, data):
        """Append rows of data to a CSV file in a specified folder."""
        filename = prefix + "_" + filename + ".csv"
        filepath = os.path.join(self.output_folder, filename)

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

        print(top_nodes)
        return top_nodes

    def load_top_closeness_centrality_nodes(self, file, top_n=20):
        filename = "close_centrality_"+file + ".txt"
        filepath = os.path.join(self.centrality_folder, filename)  # Full path to file
        # Read the edgelist from the file and add edges to the graph
        with open(filepath, "r") as file:
            lines = file.readlines()  # Read all lines
        top_nodes = [line.split("\t")[0] for line in lines[1:top_n+1] if line.strip()]  # Skip header and process non-empty lines

        print(top_nodes)
        return top_nodes

    def top_betweenness_centrality_nodes(self, G, top_n=20):
        """Find the top N nodes with the highest betweenness centrality in the graph."""
        # Calculate betweenness centrality for all nodes
        centrality = nx.betweenness_centrality(G)

        """
        mapping = {node: idx for idx, node in enumerate(G.nodes())}

        # Relabel the nodes using the mapping
        G_nx_int = nx.relabel_nodes(G, mapping)

        g = Graph(edges=list(G_nx_int.edges()))

        # Calculate closeness centrality
        centrality = g.betweenness()

        top_cent_nodes = sorted(enumerate(centrality), key=lambda x: x[1], reverse=True)[:top_n]
        """
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


    def top_closeness_centrality_nodes(self, G, top_n=20):
        """Find the top N nodes with the highest closeness centrality in the graph."""
        # Calculate closeness centrality for all nodes
        centrality = nx.closeness_centrality(G)

        """
        mapping = {node: idx for idx, node in enumerate(G.nodes())}

        # Relabel the nodes using the mapping
        G_nx_int = nx.relabel_nodes(G, mapping)

        g = Graph(edges=list(G_nx_int.edges()))

        # Calculate closeness centrality
        centrality = g.closeness()
        centrality = [0 if math.isnan(c) else c for c in centrality]
        top_cent_nodes = sorted(enumerate(centrality), key=lambda x: x[1], reverse=True) #[:top_n]
        """
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

    # Example usage
    # Assuming G is the graph you created earlier
    #top_20_nodes = top_closeness_centrality_nodes(G, top_n=20)

    # Example usage
    # Assuming G is the graph you created earlier
    #top_20_nodes = top_betweenness_centrality_nodes(G, top_n=20)


    def set_base_cent_centrality_nodes(self, G, top_n=20):
        self.base_top_k_bet_cen = self.top_betweenness_centrality_nodes(G, top_n)
        self.base_top_k_bet_cen = [ int(node) for node in self.base_top_k_bet_cen ]
        self.base_top_k_close_cen = self.top_closeness_centrality_nodes(G, top_n)
        self.base_top_k_close_cen = [int(node) for node in self.base_top_k_close_cen]

    def load_base_cent_centrality_nodes(self, file, top_n=20):
        self.base_top_k_bet_cen = self.load_top_betweenness_centrality_nodes(file, top_n)
        self.base_top_k_bet_cen = [int(node) for node in self.base_top_k_bet_cen]
        self.base_top_k_close_cen = self.load_top_closeness_centrality_nodes(file, top_n)
        self.base_top_k_close_cen = [int(node) for node in self.base_top_k_close_cen]

    def calculate_jaccard_index(self, G, filename, method, percent, top_n=20):
        print(f'percet {percent}')
        for centrality in self.centralities:
            if centrality == "betweenness":
                top_k_bet_cen = self.top_betweenness_centrality_nodes(G, top_n)
                jac_index = len(list(set(top_k_bet_cen) & set(self.base_top_k_bet_cen))) / len(
                    list(set(top_k_bet_cen) | set(self.base_top_k_bet_cen)))
                data = [
                    {"Method": method, "edge removal percentage": percent, "jaccard index": jac_index},
                ]
                self.write_values(filename, centrality, data)
                """
                print("betweenness")
                print(list(set(top_k_bet_cen)))
                print(list(set(self.base_top_k_bet_cen)))
                print(list(set(top_k_bet_cen) & set(self.base_top_k_bet_cen)))
                print(len(list(set(top_k_bet_cen) & set(self.base_top_k_bet_cen))))
                print(list(set(top_k_bet_cen) | set(self.base_top_k_bet_cen)))
                print(len(list(set(top_k_bet_cen) | set(self.base_top_k_bet_cen))))
                print(f'Jaccard index : {jac_index}')
                """
            if centrality == "closeness":
                top_k_close_cen = self.top_closeness_centrality_nodes(G, top_n)
                jac_index = len(list(set(top_k_close_cen) & set(self.base_top_k_close_cen))) / len(
                    list(set(top_k_close_cen) | set(self.base_top_k_close_cen)))
                data = [
                    {"Method": method, "edge removal percentage": percent, "jaccard index": jac_index},
                ]
                self.write_values(filename, centrality, data)
                """
                print("closeness")
                print(list(set(top_k_close_cen)))
                print(list(set(self.base_top_k_close_cen)))
                print(list(set(top_k_close_cen) & set(self.base_top_k_close_cen)))
                print(len(list(set(top_k_close_cen) & set(self.base_top_k_close_cen))))
                print(list(set(top_k_close_cen) | set(self.base_top_k_close_cen)))
                print(len(list(set(top_k_close_cen) | set(self.base_top_k_close_cen))))
                print(f'Jaccard index : {jac_index}')
                """






