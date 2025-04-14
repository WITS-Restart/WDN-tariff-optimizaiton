import random

from epanet_env import WaterNetworkEnv, gaussian_pattern, square_pattern
from agent import DQNAgent
from tqdm import tqdm
from random import randint
#import matplotlib.pyplot as plt



#!VD! erano commentati nel nostro ambiente precedente
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

import numpy as np
np.set_printoptions(precision=8)

#!VD! new
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#!VD! non c'era nel nostro ambiente precedente
from py_epanet_massive_v1_2 import simulation_step, get_nodes

#!VD! aggiungiamo noi
import csv
import os
import time
#!VD3!
import json
import numpy as np
import pandas as pd
from py_epanet_massive_v1_2 import get_lookup_table
lookup_csv_filename = "lookup_table_100k.csv"

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)  # np.int64 -> int
    elif isinstance(obj, (np.floating, float)):
        return float(obj)  # np.float64 -> float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # np.array -> lista
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")  # DataFrame -> lista di dizionari
    elif isinstance(obj, pd.Series):
        return obj.tolist()  # Series -> lista
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}  # Converti chiavi e valori
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(v) for v in obj]  # Converti elementi nelle liste/tuple/set
    else:
        return obj  # Altri tipi rimangono invariati
def save_lookup_table():
    lookup_table = get_lookup_table()
    with open(lookup_csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patterns", "df"])
        for patterns, df in lookup_table.items():
            #writer.writerow([json.dumps(patterns), df.to_dict()])
            # Convertire tutti i tipi numpy in tipi Python standard
            df_dict = df.to_dict()
            #df_serializable = {k: {k2: (int(v2) if isinstance(v2, np.integer) else v2) for k2, v2 in v.items()} for k, v
            #                   in df_dict.items()}
            df_serializable = convert_to_serializable(df_dict)
            writer.writerow([json.dumps(patterns), json.dumps(df_serializable)])
    print(f" Lookup Table salvata in {lookup_csv_filename}")




num_clients = 23
num_patterns = 3

env = WaterNetworkEnv()
env.reset()
#env.set_tariffs(gaussian_pattern(3, 1, norm="max"))
env.set_tariffs(square_pattern(9, 1, 12))








#!VD!2test_________________
# Imposta tutti i pattern uguali (es. pattern0)
#env.patterns = [0] * len(env.patterns)

#patterns = [0, 1, 3]  # I tre tipi di pattern
#pattern_assignment = np.random.choice(patterns, num_clients)
#env.patterns = pattern_assignment.tolist()

#csv_filename = "test_pattern0_results.csv"
#if not os.path.exists(csv_filename):
#    with open(csv_filename, mode="w", newline="") as file:
#        writer = csv.writer(file)
#        writer.writerow(["nodo", "acqua_soddisfatta", "acqua_non_soddisfatta"])
# Simulazione senza training
#env.step(0, 3)  # Esegue uno step per aggiornare lo stato
# Calcola acqua soddisfatta e non soddisfatta per ogni nodo
#satisfied_water = env.demandv  # Acqua soddisfatta per ogni nodo
#dissatisfied_water = env.bdemands - env.demandv  # Acqua non soddisfatta per ogni nodo
# Scrivi i risultati nel CSV
#with open(csv_filename, mode="a", newline="") as file:
#    writer = csv.writer(file)
#    for i in range(len(env.nodes)):  # Itera su tutti i nodi
#        writer.writerow([i, satisfied_water[i], dissatisfied_water[i]])
#print(f"Test completato! Risultati salvati in {csv_filename}")
#
# total_satisfied = sum(satisfied_water)
# total_dissatisfied = sum(dissatisfied_water)
# patterns_count = [env.patterns.count(0), env.patterns.count(1), env.patterns.count(3)]
#
# print(f"\n RISULTATI TEST PATTERN 0 ")
# print(f" Acqua soddisfatta totale: {float(total_satisfied.sum()):.2f}")
# print(f" Acqua non soddisfatta totale: {float(total_dissatisfied.sum()):.2f}")
# print(f" Conteggio pattern usati: {patterns_count} (Pattern0={patterns_count[0]}, Pattern1={patterns_count[1]}, Pattern3={patterns_count[2]})\n")
# exit(0)
#!VD!2test_________________








#!VD! prima non chiama device #agent = DQNAgent(num_clients, num_patterns)
#agent = DQNAgent(num_clients, num_patterns, device=device)

agent = DQNAgent(num_clients, num_patterns)
costs = []


#!VD! non c'era nel nostro ambiente precedente
#plotlosses = PlotLosses(groups={'Cost': ['cost']}, outputs=[MatplotlibPlot()])


#!VD!
lap_time = time.time()
start_time = time.time()
csv_filename = "training_results_100k.csv"
# Scrivere l'intestazione se il file non esiste
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_episode", "cost", "satisfied_water", "dissatisfied_water", "patterns", "tariff", "pattern0", "pattern1", "pattern3", "satisfied_water_sum", "dissatisfied_water_sum", "tariff0","tariff1","lap_time","time"])

#!VD!2
#csv_pressures_filename = "pressures.csv"
 #if not os.path.exists(csv_pressures_filename):
 #   with open(csv_pressures_filename, mode="w", newline="") as file2:
 #       writer2 = csv.writer(file2)
 #       # Creiamo l'intestazione: "episodio" + 24 colonne per le pressioni
 #       header = ["episode","node"] + [f"pressure_{i}" for i in range(24)]
 #       writer2.writerow(header)





for episode in tqdm(range(100000)):

    #!VD! nell'ambiente precedente
    print(episode)
    new_time = time.time() - lap_time
    print(new_time)
    lap_time = time.time()

    #env.set_tariffs(1 - gaussian_pattern(randint(0, 23), 1, norm="max"))
    if random.random()>0.5:
        env.set_tariffs(square_pattern(9, 1, 12))
    else:
        env.set_tariffs(square_pattern(21, 1, 6))

    cost = agent.train(env, num_clients, batch_size=32)
    costs.append(cost)
    #!VD! print(const)


    #!VD!2
    #pressures = env.state  #shape (23, 24) --> 23 clienti per 24 pressioni
    #print(f"Episodio {episode}: Pressioni = {pressures}")
    #print(f"Shape delle pressioni: {pressures.shape}")  # Controlla la dimensione
    # Salviamo una riga per ogni nodo
    #with open(csv_pressures_filename, mode="a", newline="") as file2:
    #    writer2 = csv.writer(file2)
    #    for i in range(23):  # Iteriamo su tutti i nodi
    #        writer2.writerow([episode, i] + pressures[i].tolist())  # Salviamo episodio, nodo e 24 pressioni



    #!VD! la parte su csv
    # .csv n_episodio | costo | acqua soddisfatta | acqua non soddisfatta | numero di nodi che hanno selezionato il pattern0,pattern1,pattern3
    satisfied_water = sum(env.demandv)
    dissatisfied_water = sum(env.bdemands - env.demandv)
    patterns = [env.patterns.count(0), env.patterns.count(1), env.patterns.count(3)]
    pattern_list = env.patterns
    satisfied_water_sum = sum(satisfied_water)
    dissatisfied_water_sum = sum(dissatisfied_water)
    tariff_selected = env.tariffs.astype(int)

    #!VD!2
    #tariff0 e tariff1 per adesso sfrutta la somma
    tariff0 = 1 if tariff_selected.sum() == 12 else 0
    tariff1 = 1 if tariff_selected.sum() == 6 else 0
    total_time = time.time() - start_time

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, cost, satisfied_water, dissatisfied_water, pattern_list, tariff_selected, *patterns, satisfied_water_sum, dissatisfied_water_sum, tariff0, tariff1, new_time, total_time])

    #!VD3!
    #if episode % 1000 == 0:
    #    save_lookup_table()

    #!VD! erano commentat nel nostro ambiente precedente
    #plotlosses.update({'cost': cost})
    #plotlosses.send()

    if episode % 10000 == 0:
        agent.learning_rate /= 2




#!VD!3
#save_lookup_table()


#!VD! da qua in giu nuovo rispetto a prima
import torch

agent.model(torch.tensor([1]), torch.tensor([i for i in range(23) if i != 1]), torch.randn(23, 24), torch.randn(1, 24))


env.step(1, 1)
#plt.plot(env.bdemands[1])

for i in range(len(env.nodes)):
    env.step(i, 1)


# %%
env.step(1, 1)[1]
# %%
env.bdemands[1]
# %%
env.bdemands[1] - env.demandv[1]
# %%
env.state