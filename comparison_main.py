
from simulate import AttackSimulator
from metrics import Jaccard_Index

jac_Index = Jaccard_Index()

files = jac_Index.get_file_names()

top_n = 20

for file in files:
    attack_simulator = AttackSimulator(jac_Index, file)
    attack_simulator.build_graph_from_edgelist(top_n)
    for _ in range(1):
        attack_simulator.attack()

