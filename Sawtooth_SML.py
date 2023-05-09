# %% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, dates
import seaborn as sns

sns.set_theme(context="paper",style="ticks",palette="colorblind")

#%% Loading in Data
sophie80df = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophie80df['Date_UTC'] = pd.to_datetime(sophie80df['Date_UTC'])
sophie80df = sophie80df[sophie80df['Date_UTC'].between('1996','2021')].reset_index(drop=True)
sophie80df['Delbay'] = pd.to_numeric(sophie80df['Delbay'],errors='coerce')

smedf = pd.read_csv("Data/SuperMAGData.csv")
smedf['Date_UTC'] = pd.to_datetime(smedf['Date_UTC'])

sawtoothdf = pd.read_csv("Data/sawtooth_events.txt", delim_whitespace=True, header=None, names=['Year','Month','Day','Hour','Minute','Second', 'Value'])
sawtoothdf['Date_UTC'] = pd.to_datetime(sawtoothdf[['Year','Month','Day','Hour','Minute','Second']])
sawtoothdf = sawtoothdf.drop(columns=['Year','Month','Day','Hour','Minute','Second'])
sawtoothdf = sawtoothdf[['Date_UTC', 'Value']]

smllt = pd.read_csv("Data/SML_LT_1999-2002.txt")
smllt['Date_UTC'] = pd.to_datetime(smllt['Date_UTC'])

#%% Data Subselection & Plotting
tstart = pd.to_datetime("1999-02-18 03:00")
duration = pd.Timedelta(hours=17)

tend = pd.to_datetime(tstart + duration)

datetimes = smedf[smedf['Date_UTC'].between(tstart, tend)]['Date_UTC']
sml = smedf[smedf['Date_UTC'].between(tstart, tend)]['SML']
smu = smedf[smedf['Date_UTC'].between(tstart, tend)]['SMU']

phasesindices = sophie80df[sophie80df['Date_UTC'].between(tstart, tend)].index.to_numpy()
phasesindices = np.concatenate(([phasesindices[0]-1],phasesindices,[phasesindices[-1]+1]))

sophieslice = sophie80df.iloc[phasesindices]
sawtoothslice = sawtoothdf[sawtoothdf['Date_UTC'].between(tstart, tend)]

smlltslice = smllt[smllt['Date_UTC'].between(tstart, tend)]

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=600)

ax.plot(datetimes, sml,label="SML")
ax.plot(datetimes, smu,label="SMU")
ax.set_xlim(tstart,tend)

ax.plot([],[],color='green',alpha=0.2,label="Growth")
ax.plot([],[],color='blue',alpha=0.2,label="Recovery")
ax.plot([],[],color='red',alpha=0.2,label="Expansion")
ax.plot([],[],color='k',alpha=0.2,label="Convection")
ax.plot([],[],color='k',linestyle='--',alpha=0.8,label="Sawtooth")

for index, row in sophieslice.iloc[:-1].iterrows():
    if sophieslice.loc[index]['Phase'] == 1: # Growth
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='green', alpha=0.2)
    if sophieslice.loc[index]['Phase'] == 2 and sophieslice.loc[index]['Flag'] == 0: # Expansion
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='red', alpha=0.2)
    if sophieslice.loc[index]['Phase'] == 3 and sophieslice.loc[index]['Flag'] == 0: # Recovery
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='blue', alpha=0.2)
    if sophieslice.loc[index]['Flag'] == 1: # Convection
        ax.axvspan(sophieslice.loc[index]['Date_UTC'], sophieslice.loc[index+1]['Date_UTC'], facecolor='k', alpha=0.2)

mesh = ax1.pcolormesh(datetimes,np.arange(24),smlltslice.values[:,1:].astype(float).T)
fig.colorbar(mesh, ax=ax1, label="SML LT (nT)",location='top')
ax1.set_ylim(0,23)
ax1.set_yticks(np.arange(0,24,6))
ax1.set_xlabel("Date (UTC)")
ax1.set_ylabel("MLT")

ax.xaxis.set_minor_locator(dates.HourLocator(interval=1))
ax.xaxis.set_major_locator(dates.HourLocator(interval=4))
ax.grid(which='major', axis='both', alpha=1)
ax.xaxis.set_major_formatter(dates.DateFormatter("%Y/%m/%d\n%H:%M"))
ax.set_ylabel("SML (nT)")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False,fontsize='small')

for index, row in sawtoothslice.iterrows():
    ax.axvline(sawtoothslice.loc[index]['Date_UTC'], color='k', linestyle='--', alpha=0.8)
    ax1.axvline(sawtoothslice.loc[index]['Date_UTC'], color='k', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.show()

# %%
