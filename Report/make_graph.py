import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

win_times = []
average_turns = []

import os

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

for i in range(1, 10):
    csv_path = f'/Users/owl/UTokyo/3S/ids_prog/submarine-py/Report/result/trial{i}/game_stats.csv'
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        continue
    
    try:
        df = pd.read_csv(csv_path)
        win_time = df['Result'].value_counts().get("win", 0)
        average_turn = df['Turns'].mean()
        win_times.append(win_time)
        average_turns.append(average_turn)
        plt.plot(df['Turns'], label=f'Game {i}')
        plt.savefig(f'./data/graph_{i}.png')
        plt.clf()
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        continue

# Check if we have any data before calculating averages
if len(average_turns) > 0:
    Overall_win_rate = sum(win_times) / (len(average_turns) * 100)  # Assuming 100 games per trial
    Overall_average_turns = sum(average_turns) / len(average_turns)
    print(f'Overall win rate: {Overall_win_rate:.2%}')
    print(f'Overall average turns: {Overall_average_turns:.2f}')
else:
    print("No valid data files found. Please check the file paths and ensure the CSV files exist.")
    print("Expected directory structure: /Users/owl/UTokyo/3S/ids_prog/submarine-py/Report/result/trial_X/game_stats.csv")