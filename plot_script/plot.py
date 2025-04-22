

import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np



# Load file CSV
file_path = "training_results.csv"
df = pd.read_csv(file_path)

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
# Graph 1: Cost per episode trend
#add rolling_avg
window = 500
rolling_avg = df['cost'].rolling(window=window, min_periods=1).mean()
#
plt.figure(figsize=(10, 5))
plt.plot(df['n_episode'], df['cost'], label='Cost per episode', color='b')
plt.plot(df['n_episode'], rolling_avg, label=f'Moving average (window={window})', color='r')
plt.xlabel('Episode')
plt.ylabel('Cost')
plt.title('Cost per episode trend')
plt.legend()
plt.grid()
#plt.show()

#save
output_filename = "plot_" + "Cost" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")
plt.close()

exit(0)


# Filter data by tariff0 == 1
df_tariff0 = df[df["tariff0"] == 1]
rolling_avg_tariff0 = df_tariff0["cost"].rolling(window=window, min_periods=1).mean()

# Creating and saving the graph for tariff0 == 1
plt.figure(figsize=(10, 5))
plt.plot(df_tariff0["n_episode"], df_tariff0["cost"], label="Cost per episode", color="b")
plt.plot(df_tariff0["n_episode"], rolling_avg_tariff0, label=f"Moving average (window={window})", color="r")
plt.xlabel("Episode")
plt.ylabel("Cost")
plt.title("Cost per episode trend (tariff0 = 1)")
plt.legend()
plt.grid()
plt.savefig("./fig/plot_Cost_tariff0.png", dpi=300, bbox_inches="tight")
plt.close()

# Filter data by tariff1 == 1
df_tariff1 = df[df["tariff1"] == 1]
rolling_avg_tariff1 = df_tariff1["cost"].rolling(window=window, min_periods=1).mean()

# Creating and saving the graph for tariff1 == 1
plt.figure(figsize=(10, 5))
plt.plot(df_tariff1["n_episode"], df_tariff1["cost"], label="Cost per episode", color="b")
plt.plot(df_tariff1["n_episode"], rolling_avg_tariff1, label=f"Moving average (window={window})", color="r")
plt.xlabel("Episode")
plt.ylabel("Cost")
plt.title("Cost per episode trend (tariff1 = 1)")
plt.legend()
plt.grid()
plt.savefig("./fig/plot_Cost_tariff1.png", dpi=300, bbox_inches="tight")
plt.close()



exit(0)


##################################
# Graph 2: Distribution of the chosen patterns over time
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

output_filename = "plot_" + "pattern" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")


#############################
# Graph 3: Trend of satisfied and dissatisfied water
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

output_filename = "plot_" + "water" + ".png"
plt.savefig( "./fig/" + output_filename, dpi=300, bbox_inches="tight")


