import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


#!VD! qua è stato tolto class AttentionModel(nn.Module) e forward(...)

class SimpleModel(nn.Module):
    def __init__(self, num_users, num_patterns):
        super(SimpleModel, self).__init__()


        #new agent 2
        self.emb_id = nn.Embedding(num_users + 1, 16)  # add 1 to handle the next_id of the last id
        #self.mlp_id = nn.Sequential(
        #    nn.Linear(1, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 16)
        #)

        self.mlp_pressures = nn.Sequential(
            nn.Linear(24 * num_users, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.mlp = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_patterns)
        )
    
    def forward(self, user_id, pressures, tariffs):
        #new agent 2
        user_id_feat = self.emb_id(user_id)
        #user_id = user_id.unsqueeze(1).float()
        #user_id_feat = self.mlp_id(user_id)

        #!VD! qua prima veniva fatto pressures = pressures.reshape(1, -1)
        #!VD!2# Normalizzazione Min-Max delle pressioni con range [0, 1]
        pressures_norm = (pressures - 0) / (90 - 0)  # Min = 0, Max = 45 (trovati su questa rete!!!)
        pressures_feat = self.mlp_pressures(pressures_norm)  #(pressures)

        # !VD!2!
        combined_input = torch.cat([user_id_feat, pressures_feat, tariffs], dim=1)
        #print("User ID features:", user_id_feat)
        #print("Pressures features:", pressures_feat)
        #print("Tariffs:", tariffs)
        #print("Final input shape:", combined_input.shape)
        #print("User ID features after MLP:", user_id_feat)
        #print(f"User ID range: min={user_id.min().item()}, max={user_id.max().item()}")
        #print(f"Tariffs range: min={tariffs.min().item()}, max={tariffs.max().item()}")

        output = self.mlp(torch.cat([user_id_feat, pressures_feat, tariffs], dim=1))

        return output


class DQNAgent:
    def __init__(self, num_clients, num_patterns, learning_rate=0.001, device=torch.device("cpu")):  # !VD! nuovo device
        self.num_patterns = num_patterns
        self.clients = [i for i in range(num_clients)]
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = device

        # !VD5#
        self.model = SimpleModel(num_clients, num_patterns).to(self.device)
        self.target_model = SimpleModel(num_clients, num_patterns).to(self.device)
        self.update_target_network()  # Inizializza la target network con gli stessi pesi

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.loss.to(self.device)
        # !VD5#
        self.train_step = 0
        self.target_update_freq = 500

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #!VD! qualche cambiamento anche qua in questa def
    def predict_action(self, id, state, tariffs):
        if np.random.rand() <= self.epsilon:
            # Esplora: scegli un'azione casuale
            return random.randint(0, self.num_patterns - 1)
        else:
            # Sfrutta: scegli l'azione con Q massimo
            q_values = self.model(id, torch.FloatTensor(state).to(self.device), tariffs.reshape(-1, 24)).cpu().detach().numpy()
            return np.argmax(q_values)

        #return action if action != 2 else 3 #cambiare se cambiano i pattern

    def remember(self, id, state, tariffs, action, reward, next_id, next_state, done):
        self.memory.append((id, state, tariffs, action, reward, next_id, next_state, done))


    #!VD! cambiamento anche qua in questa def
    #!VD! qua avevo messo prima la patch:
        # print(action, target, target_f.shape)
        # da controllare xke va fuori il limite!!!
        #if action >= target_f.shape[1]:
            #print(f"Warning: action {action} is out of bounds (max index {target_f.shape[1] - 1}). Clamping to valid range.")
            #action = target_f.shape[1] - 1
        #target_f[0][action] = target
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        ids = torch.tensor(np.array([id for id, _, _, _, _, _, _, _ in minibatch])).to(self.device)
        states = torch.FloatTensor(np.array([s for _, s, _, _, _, _, _, _ in minibatch])).reshape(-1, 24 * len(self.clients)).to(self.device)
        tariffs = torch.FloatTensor(np.array([t for _, _, t, _, _, _, _, _ in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([a for _, _, _, a, _, _, _, _ in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([r for _, _, _, _, r, _, _, _ in minibatch])).to(self.device)
        next_ids = torch.tensor(np.array([ni for _, _, _, _, _, ni, _, _ in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, _, _, _, ns, _ in minibatch])).reshape(-1, 24 * len(self.clients)).to(self.device)
        dones = torch.FloatTensor(np.array([float(d) for _, _, _, _, _, _, _, d in minibatch])).to(self.device)
        
        q_values = self.model(ids, states, tariffs).gather(1, actions).squeeze(1)
        
        #next_q_values = self.model(next_ids, next_states, tariffs).detach().max(1)[0]
        #!VD5#
        next_q_values = self.target_model(next_ids, next_states, tariffs).detach().max(1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #!VD5!
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

    def train(self, env, num_clients, batch_size):
        state = env.reset()
        total_cost = 0

        # qui posso eseguire epanet con tutti i pattern generati dal modello
        # e poi calcolare il costo come prezzo pagato da ogni cliente + una penalità
        # dipendente da quanta acqua non è stata soddisfatta (base demand - demand value)

        # Problema: mi serve uno stato rappresentativo della situazione
        # altrimenti il modello non impara

        for client_idx in range(num_clients):
            action = self.predict_action(torch.tensor([client_idx]).to(self.device), state.reshape(1, -1), torch.FloatTensor(env.tariffs).to(self.device))

            next_state, cost = env.step(client_idx, action)
            total_cost += cost

            reward = -cost

            next_id = client_idx + 1 # cambiare se si cambia la modalità di estrazione dei clienti #!VD! vedere

            self.remember(client_idx, state, env.tariffs, action, reward, next_id, next_state, done=(client_idx == num_clients - 1))

            state = next_state

        # problema: non esiste uno stato
        # possibili features potrebbero essere:
        # numero di nodi tra me e il reservoir più vicino
        # pressione al nodo ----> che però è influenzata dall'assorbimento degli altri nodi
        # 
        # inoltre il sistema non sarebbe comunque adattabile perchè il training dovrebbe ricominciare da capo
        # ad esempio se si aggiunge o rimuove un cliente o un pattern

            #!VD5#
            self.replay(batch_size)
        #self.replay(batch_size)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return total_cost
    
    def eval(self, env, num_clients):
        # TODO: verificare che funzioni correttamente
        state = env.reset()
        total_cost = 0

        patterns = []

        for client_idx in range(num_clients):
            client_idx = torch.tensor([client_idx])

            action = self.predict_action(client_idx, state.reshape(1, -1), torch.FloatTensor(env.tariffs))
            
            patterns.append(action)

            next_state, cost = env.step(client_idx, action)
            total_cost += cost

            state = next_state

        return total_cost, patterns
    
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))