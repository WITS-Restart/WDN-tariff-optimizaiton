import numpy as np
import gym

from py_epanet_massive_v1_2 import simulation_step, get_nodes

class WaterNetworkEnv(gym.Env):
    def __init__(self):
        reservoir = 7384
        self.nodes = get_nodes()
        index = np.argwhere(self.nodes==reservoir)
        self.nodes = np.delete(self.nodes, index)

        self.num_users = len(self.nodes)

        self.patterns = [-1 for _ in range(self.num_users)] #!VD! prima era zero
        
    def set_tariffs(self, tariffs):
        self.tariffs = tariffs

    def reset(self):
        self.patterns = [-1 for _ in range(self.num_users)] #!VD! prima era zero
        df = simulation_step(self.patterns)
        self._parse_state(df)
        return self.state

    def get_state(self):
        return self.state
    
    def _parse_state(self, df):
        pressures = []
        self.bdemands = []
        self.demandv = []
        for node in self.nodes:
            pressures.append(df[df['nodeID'] == int(node)]['pressure_value'].to_numpy())
            self.bdemands.append(df[df['nodeID'] == int(node)]['base_demand'].to_numpy())
            self.demandv.append(df[df['nodeID'] == int(node)]['demand_value'].to_numpy())
        self.state = np.array(pressures, dtype=np.float64)
        self.bdemands = np.array(self.bdemands, dtype=np.float64)
        self.demandv = np.array(self.demandv, dtype=np.float64)

    def step(self, i, pattern):
        self.patterns[i] = pattern if pattern != 2 else 3 #cambiare se cambiano i pattern #!VD!

        not_satisfied_water_before = sum(sum(self.bdemands - self.demandv))

        self._parse_state(simulation_step(self.patterns))

        not_satisfied_water_after = sum(sum(self.bdemands - self.demandv))

        dv = self.demandv[i]
        #print(dv, self.tariffs)

        paid_price = sum(dv * self.tariffs)

        #!VD! prima era così cost = paid_price + 10 * (not_satisfied_water_after - not_satisfied_water_before)
        cost = 100 * paid_price + 1000 * (not_satisfied_water_after - not_satisfied_water_before)

        return self.state, cost
    
def gaussian_pattern(peak_hour, total_consumption, std_dev=3, norm="sum"):
    # if norm = "sum", the total consumption equals total_consumption
    # if norm = "max", the peak consumption will be equal to total_consumption
    hours = np.arange(24)
    distance = np.minimum(np.abs(hours - peak_hour), 24 - np.abs(hours - peak_hour))
    gaussian = np.exp(-0.5 * (distance / std_dev) ** 2)
    if norm == "sum":
        gaussian /= gaussian.sum()
    elif norm == "max":
        gaussian /= gaussian.max()
    return gaussian * total_consumption

def square_pattern(center_hour, total_consumption, duration=6):
    square = np.zeros(24)
    duration_pattern =np.ones(duration)

    left_index = center_hour - int(duration/2)

    if left_index<0:
        left_index=0
    if left_index+duration>23:
        left_index=24-duration

    square[left_index:left_index+duration] += duration_pattern
    return square * total_consumption