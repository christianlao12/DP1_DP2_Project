# %% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, dates
import seaborn as sns

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)

#%% Loading in Data
sophie80df = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophie80df['Date_UTC'] = pd.to_datetime(sophie80df['Date_UTC'])
sophie80df = sophie80df[sophie80df['Date_UTC'].between('1996','2021')].reset_index(drop=True)
sophie80df['Delbay'] = pd.to_numeric(sophie80df['Delbay'],errors='coerce')

smedf = pd.read_csv("Data/SuperMAGData.csv")
smedf['Date_UTC'] = pd.to_datetime(smedf['Date_UTC'])

#%% Plotting
tstart = pd.to_datetime("2002-04-18")
tend = pd.to_datetime("2002-04-19")

datetimes = smedf[smedf['Date_UTC'].between(tstart, tend)]['Date_UTC']
sml = smedf[smedf['Date_UTC'].between(tstart, tend)]['SML']

smeindices = sophie80df[sophie80df['Date_UTC'].between(tstart, tend)].index.to_numpy()
smeindices = np.concatenate(([smeindices[0]-1],smeindices,[smeindices[-1]+1]))

sophieslice = sophie80df.iloc[smeindices]
cm = 1/2.54
fig, ax = plt.subplots(figsize=(18*cm,6*cm),dpi=300)
ax.plot(datetimes, sml)
ax.set_xlim(tstart,tend)

for index, row in sophieslice.iloc[:-1].iterrows():
    if sophieslice.loc[index]['Phase'] == 1: # Growth
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='green', alpha=0.2)
    if sophieslice.loc[index]['Phase'] == 2 and sophieslice.loc[index]['Flag'] == 0: # Expansion
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='red', alpha=0.2)
    if sophieslice.loc[index]['Phase'] == 3 and sophieslice.loc[index]['Flag'] == 0: # Recovery
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='blue', alpha=0.2)
    if sophieslice.loc[index]['Flag'] == 1: # Convection
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='k', alpha=0.2)
ax.set_ylim(top=0)
ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d\n%H:%M"))
ax.set_xlabel("Date (UTC)")
ax.set_ylabel("SML (nT)")
plt.tight_layout()
plt.show()

# %%
