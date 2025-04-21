from simulate import AttackSimulator
from metrics import Jaccard_Index
import csv
import os

def write_header(fieldnames):
    """Write the header to a CSV file in a specified folder."""
    output_folder = "output"
    filename = "precision_recall" + ".txt"
    filepath = os.path.join(output_folder, filename)  # Full path to file
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

def write_values(data):
    """Append rows of data to a CSV file in a specified folder."""
    output_folder = "output"
    filename = "precision_recall" + ".txt"
    filepath = os.path.join(output_folder, filename)  # Full path to file
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writerows(data)

if __name__=="__main__":
    fieldnames = ["Name", "Precision", "Recall"]
    write_header(fieldnames)

    jac_Index = Jaccard_Index()

    files = jac_Index.get_file_names()
    top_n = 20

    for file in files:
        attack_model = AttackSimulator(jac_Index, file)
        attack_model.build_graph_from_edgelist(top_n)
        for _ in range(1):
            precision, recall = attack_model.calculate_precision_recall(top_n)
            data = [
                {"Name": file, "Precision": precision, "Recall": recall},
            ]
            write_values(data)
