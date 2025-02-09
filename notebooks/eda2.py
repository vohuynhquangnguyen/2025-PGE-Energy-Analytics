import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("../data/raw/training.xlsx", header=0)

df['Year'] = df['Year'] + 2019
df['Hour'] = df['Hour'] - 1
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']].astype(str).apply(lambda x: f"{int(x[0]):04d}-{int(x[1]):02d}-{int(x[2]):02d} {int(x[3]):02d}:00:00", axis=1))


#TODO: Shows all sites temps in one plot
plt.figure(figsize=(15, 8))
for i in range(1, 6):
    plt.plot(df['datetime'], df[f'Site-{i} Temp'], label=f'Site-{i} Temp')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Hourly Temperature Time Series for Each Site')
plt.legend()
plt.tight_layout()
plt.show()

#TODO: Shows each sites temp in its own plot
for i in range(1, 6):
    plt.figure(figsize=(15, 8))
    plt.plot(df['datetime'], df[f'Site-{i} Temp'], label=f'Site-{i} Temp')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Hourly Temperature Time Series for Each Site')
    plt.legend()
    plt.tight_layout()
    plt.show()

#TODO: Shows all sites GHI in one plot
#removes values of 0 from being plotted
for i in range(1, 6):
    df = df[df[f'Site-{i} GHI'] != 0]
plt.figure(figsize=(15, 8))
for i in range(1, 6):
    plt.plot(df['datetime'], df[f'Site-{i} GHI'], label=f'Site-{i} GHI')
plt.xlabel('Time')
plt.ylabel('GHI')
plt.title('Hourly GHI Time Series for Each Site')
plt.legend()
plt.tight_layout()
plt.show()

#TODO: Shows each sites GHI in its own plot
for i in range(1, 6):
    plt.figure(figsize=(15, 8))
    plt.plot(df['datetime'], df[f'Site-{i} GHI'], label=f'Site-{i} GHI')
    plt.xlabel('Time')
    plt.ylabel('GHI')
    plt.title('Hourly GHI Time Series for Each Site')
    plt.legend()
    plt.tight_layout()
    plt.show()



