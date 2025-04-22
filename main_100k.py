import numpy as np
np.set_printoptions(precision=8)
import random
import csv
import os
import time
from epanet_env import WaterNetworkEnv, gaussian_pattern, square_pattern
from py_epanet_massive_v1_2 import simulation_step, get_nodes
from agent import DQNAgent
from tqdm import tqdm
from random import randint
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import json
import pandas as pd
from py_epanet_massive_v1_2 import get_lookup_table
lookup_csv_filename = "lookup_table_100k.csv"




num_clients = 23
num_patterns = 3

env = WaterNetworkEnv()
env.reset()
env.set_tariffs(square_pattern(9, 1, 12))

agent = DQNAgent(num_clients, num_patterns)
costs = []

lap_time = time.time()
start_time = time.time()
csv_filename = "training_results_100k.csv"
#
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_episode", "cost", "satisfied_water", "dissatisfied_water", "patterns", "tariff", "pattern0", "pattern1", "pattern3", "satisfied_water_sum", "dissatisfied_water_sum", "tariff0","tariff1","lap_time","time"])



for episode in tqdm(range(100000)):

    print(episode)
    new_time = time.time() - lap_time
    print(new_time)
    lap_time = time.time()

    if random.random()>0.5:
        env.set_tariffs(square_pattern(9, 1, 12))
    else:
        env.set_tariffs(square_pattern(21, 1, 6))

    cost = agent.train(env, num_clients, batch_size=32)
    costs.append(cost)


    satisfied_water = sum(env.demandv)
    dissatisfied_water = sum(env.bdemands - env.demandv)
    patterns = [env.patterns.count(0), env.patterns.count(1), env.patterns.count(3)]
    pattern_list = env.patterns
    satisfied_water_sum = sum(satisfied_water)
    dissatisfied_water_sum = sum(dissatisfied_water)
    tariff_selected = env.tariffs.astype(int)

    tariff0 = 1 if tariff_selected.sum() == 12 else 0
    tariff1 = 1 if tariff_selected.sum() == 6 else 0
    total_time = time.time() - start_time

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, cost, satisfied_water, dissatisfied_water, pattern_list, tariff_selected, *patterns, satisfied_water_sum, dissatisfied_water_sum, tariff0, tariff1, new_time, total_time])

    if episode % 10000 == 0:
        agent.learning_rate /= 2

