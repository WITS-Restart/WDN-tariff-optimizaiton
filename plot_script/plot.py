

import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np



# Caricare il file CSV
file_path = "training_(6)_results(new_agent2).csv" #"training_results_100k.csv"
df = pd.read_csv(file_path)

# Estrazione delle colonne
#episodes = df.iloc[:, 0]  # Numero episodio
#cost = df.iloc[:, 1]  # Costo
#tariff0 = df.iloc[:, 11]
#tariff1 = df.iloc[:, 12]
#satisfied_water = df.iloc[:, 2]  # Acqua soddisfatta
#unsatisfied_water = df.iloc[:, 3]  # Acqua non soddisfatta
#pattern_0 = df.iloc[:, 4]  # Nodi con Pattern 0
#pattern_1 = df.iloc[:, 5]  # Nodi con Pattern 1
#pattern_3 = df.iloc[:, 6]  # Nodi con Pattern 3
#tariffs = df.iloc[:, 7]  # Tariffa
#tariffs_sum = tariffs.apply(lambda x: np.sum(np.fromstring(x.strip('[]'), sep=' ')))
episodes = df['n_episode']
cost = df['cost']
satisfied_water = df['satisfied_water']
dissatisfied_water = df['dissatisfied_water']
patterns = df['patterns']
tariff = df['tariff']
pattern0 = df['pattern0']
pattern1 = df['pattern1']
pattern3 = df['pattern3']
satisfied_water_sum = df['satisfied_water_sum']
dissatisfied_water_sum = df['dissatisfied_water_sum']
tariff0 = df['tariff0']
tariff1 = df['tariff1']
lap_time = df['lap_time']
time = df['time']



##############################
# Grafico 1: Andamento del costo per episodio
#aggiungi curva rolling_avg
window = 500 #000  # Puoi modificare questo valore
rolling_avg = df['cost'].rolling(window=window, min_periods=1).mean()
#
plt.figure(figsize=(10, 5))
plt.plot(df['n_episode'], df['cost'], label='Costo per episodio', color='b')
plt.plot(df['n_episode'], rolling_avg, label=f'Media mobile (window={window})', color='r')
plt.xlabel('Episodio')
plt.ylabel('Costo')
plt.title('Andamento del costo per episodio')
plt.legend()
plt.grid()
#plt.show()

#poi da salvare le varie figure
output_filename = "plot_" + "Cost_NewAgent2_(9k)" + ".png" #"plot_" + "Cost_20000_rolling_avg1000" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")
plt.close()

exit(0)

# Filtrare i dati per tariff0 == 1
df_tariff0 = df[df["tariff0"] == 1]
rolling_avg_tariff0 = df_tariff0["cost"].rolling(window=window, min_periods=1).mean()

# Creazione e salvataggio del grafico per tariff0 == 1
plt.figure(figsize=(10, 5))
plt.plot(df_tariff0["n_episode"], df_tariff0["cost"], label="Costo per episodio", color="b")
plt.plot(df_tariff0["n_episode"], rolling_avg_tariff0, label=f"Media mobile (window={window})", color="r")
plt.xlabel("Episodio")
plt.ylabel("Costo")
plt.title("Andamento del costo per episodio (tariff0 = 1)")
plt.legend()
plt.grid()
plt.savefig("./fig/plot_Cost_tariff0_rolling_avg1000.png", dpi=300, bbox_inches="tight")
plt.close()

# Filtrare i dati per tariff1 == 1
df_tariff1 = df[df["tariff1"] == 1]
rolling_avg_tariff1 = df_tariff1["cost"].rolling(window=window, min_periods=1).mean()

# Creazione e salvataggio del grafico per tariff1 == 1
plt.figure(figsize=(10, 5))
plt.plot(df_tariff1["n_episode"], df_tariff1["cost"], label="Costo per episodio", color="b")
plt.plot(df_tariff1["n_episode"], rolling_avg_tariff1, label=f"Media mobile (window={window})", color="r")
plt.xlabel("Episodio")
plt.ylabel("Costo")
plt.title("Andamento del costo per episodio (tariff1 = 1)")
plt.legend()
plt.grid()
plt.savefig("./fig/plot_Cost_tariff1_rolling_avg1000.png", dpi=300, bbox_inches="tight")
plt.close()



exit(0)


##################################
# Grafico 2: Distribuzione dei pattern scelti nel tempo
plt.figure(figsize=(10, 5))
plt.plot(df['n_episode'], df['pattern0'], label='Pattern 0', color='r')
plt.plot(df['n_episode'], df['pattern1'], label='Pattern 1', color='g')
plt.plot(df['n_episode'], df['pattern3'], label='Pattern 3', color='b')
plt.xlabel('Episode')
plt.ylabel('Number of nodes')
plt.title('Distribution of patterns over time')
plt.legend()
plt.grid()
#plt.show()
#poi da salvare le varie figure
output_filename = "plot_" + "pattern" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")


#############################
# Grafico 3: Andamento dell'acqua soddisfatta e non soddisfatta
# Convertire le stringhe in array numerici e sommare i valori
plt.figure(figsize=(10, 5))
df['satisfied_water'] = df['satisfied_water'].apply(lambda x: np.sum(np.fromstring(x.strip('[]'), sep=' ')))
df['dissatisfied_water'] = df['dissatisfied_water'].apply(lambda x: np.sum(np.fromstring(x.strip('[]'), sep=' ')))
plt.figure(figsize=(10, 5))
#plt.plot(df['n_episode'], df['satisfied_water'], label='Satisfied water', color='c')
plt.plot(df['n_episode'], df['dissatisfied_water'], label='Water not satisfied', color='m')
plt.xlabel('Episode')
plt.ylabel('Amount of water')
plt.title('Satisfied and Dissatisfied Water Trend')
plt.legend()
plt.grid()
#plt.show()
#poi da salvare le varie figure
output_filename = "plot_" + "water" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")



### ---- # Creazione della figura con tre sottografici ---- ###
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Grafico 1 - Pattern 0
axes[0].plot(episodes, pattern_0, color='red', linestyle='-', linewidth=1.5, label="Pattern 0")
axes[0].set_ylabel("Number of nodes", fontsize=12)
axes[0].set_title("Nodes selecting Pattern 0", fontsize=14)
axes[0].legend()
axes[0].grid(True)

# Grafico 2 - Pattern 1
axes[1].plot(episodes, pattern_1, color='green', linestyle='-', linewidth=1.5, label="Pattern 1")
axes[1].set_ylabel("Number of nodes", fontsize=12)
axes[1].set_title("Nodes selecting Pattern 1", fontsize=14)
axes[1].legend()
axes[1].grid(True)

# Grafico 3 - Pattern 3
axes[2].plot(episodes, pattern_3, color='blue', linestyle='-', linewidth=1.5, label="Pattern 3")
axes[2].set_xlabel("Episode", fontsize=12)
axes[2].set_ylabel("Number of nodes", fontsize=12)
axes[2].set_title("Nodes selecting Pattern 3", fontsize=14)
axes[2].legend()
axes[2].grid(True)


# Grafico 4 - tariff
axes[3].scatter(episodes, tariffs_sum, label="tariff")
axes[3].set_xlabel("Episode", fontsize=12)
axes[3].set_ylabel("Selected tariff for episode", fontsize=12)
axes[3].set_title("Nodes selecting tariff", fontsize=14)
axes[3].legend()
axes[3].grid(True)

# Mostra
plt.tight_layout()
#plt.show()

#poi da salvare le varie figure
output_filename = "plot_" + "pattern" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")


#################
### ---- PATTERN DISTRIBUTION (PERCENTAGE) ---- ###
total_nodes = pattern_0 + pattern_1 + pattern_3
pattern_0_pct = (pattern_0 / total_nodes) * 100
pattern_1_pct = (pattern_1 / total_nodes) * 100
pattern_3_pct = (pattern_3 / total_nodes) * 100

plt.figure(figsize=(10, 5))
plt.plot(episodes, pattern_0_pct, color="red", linewidth=1, label="Pattern 0 (%)")
plt.plot(episodes, pattern_1_pct, color="green", linewidth=1, label="Pattern 1 (%)")
plt.plot(episodes, pattern_3_pct, color="blue", linewidth=1, label="Pattern 3 (%)")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Percentage of nodes", fontsize=12)
plt.title("Pattern distribution over time (percentage)", fontsize=14)
plt.legend()
plt.grid(True)
#plt.show()
#poi da salvare le varie figure
output_filename = "plot_" + "pattern_percent" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")



### ---- PATTERN SEPARATE PLOTS (PERCENTAGE) ---- ###
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(episodes, pattern_0_pct, color="red", linewidth=1)
axes[0].set_title("Pattern 0 over time (percentage)", fontsize=12)
axes[0].set_ylabel("Percentage", fontsize=10)
axes[0].grid(True)

axes[1].plot(episodes, pattern_1_pct, color="green", linewidth=1)
axes[1].set_title("Pattern 1 over time (percentage)", fontsize=12)
axes[1].set_ylabel("Percentage", fontsize=10)
axes[1].grid(True)

axes[2].plot(episodes, pattern_3_pct, color="blue", linewidth=1)
axes[2].set_title("Pattern 3 over time (percentage)", fontsize=12)
axes[2].set_xlabel("Episode", fontsize=10)
axes[2].set_ylabel("Percentage", fontsize=10)
axes[2].grid(True)

plt.tight_layout()
#plt.show()
#poi da salvare le varie figure
output_filename = "plot_" + "pattern_percent_2" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")



### ---- PATTERN BOX PLOT (Distribuzione statistica) ---- ###
#df_patterns = pd.DataFrame({"Pattern 0": pattern_0, "Pattern 1": pattern_1, "Pattern 3": pattern_3})
#plt.figure(figsize=(8, 6))
#sns.boxplot(data=df_patterns, palette=["red", "green", "blue"])
#plt.title("Pattern Distribution (Box Plot)", fontsize=14)
#plt.ylabel("Number of nodes", fontsize=12)
#plt.show()

