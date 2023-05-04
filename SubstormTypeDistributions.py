# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)

#%% Loading in SOPHIE Data
sophie80df = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophie80df['Date_UTC'] = pd.to_datetime(sophie80df['Date_UTC'])
sophie80df = sophie80df[sophie80df['Date_UTC'].between('1996','2021')].reset_index(drop=True)
sophie80df['Delbay'] = pd.to_numeric(sophie80df['Delbay'],errors='coerce')

array = np.zeros(len(sophie80df['Date_UTC']),dtype=int)
for i in range(1,len(sophie80df['Date_UTC'])-2):
    if (sophie80df.iloc[i-1]['Phase'] == 1) and (sophie80df.iloc[i]['Phase'] == 2) and (sophie80df.iloc[i]['Flag'] == 0) and (sophie80df.iloc[i+1]['Phase'] == 3) and (sophie80df.iloc[i+2]['Phase'] == 1):
        array[i] = 1 # GERG
        continue
    if (sophie80df.iloc[i-1]['Phase'] == 1) and (sophie80df.iloc[i]['Phase'] == 2) and (sophie80df.iloc[i]['Flag'] == 0) and (sophie80df.iloc[i+1]['Phase'] == 3) and (sophie80df.iloc[i+2]['Phase'] != 1):
        array[i] = 2 # GER...
        continue
    else:
        array[i] = 0
        continue
sophie80df['Isolated Type'] = array

#%% Plotting
onlyonsets_wt = np.diff(sophie80df['Date_UTC'][sophie80df['Isolated Type'] == 1])/pd.to_timedelta(1, unit='h')
onsets_extra_wt = np.diff(sophie80df['Date_UTC'][sophie80df['Isolated Type'] == 2])/pd.to_timedelta(1, unit='h')

fig, axes = plt.subplots()

sns.histplot(onlyonsets_wt,
             bins=np.arange(0,24.25,0.25),
             ax=axes, stat='percent',
             label='G|ERG EPT80: Mean: {:.2f}, Std. Dev: {:.2f}, Median: {:.2f}'.format(np.nanmean(onlyonsets_wt),np.nanstd(onlyonsets_wt),np.nanmedian(onlyonsets_wt)))
sns.histplot(onsets_extra_wt,
             bins=np.arange(0,24.25,0.25),
             ax=axes,
             stat='percent',
             label='G|ER... EPT80: Mean: {:.2f}, Std. Dev: {:.2f}, Median: {:.2f}'.format(np.nanmean(onsets_extra_wt),np.nanstd(onsets_extra_wt),np.nanmedian(onsets_extra_wt)))
axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
axes.legend(loc='upper right')
axes.set_xlabel('Waiting Time (Hours)')
axes.set_ylabel('Probability (%)')
axes.set_xlim(0,24)
plt.show()

onsetsonlysize = -sophie80df['Delbay'][sophie80df['Isolated Type'] == 1]
onsetextrasize = -sophie80df['Delbay'][sophie80df['Isolated Type'] == 2]

fig, axes = plt.subplots()

sns.histplot(onsetsonlysize,
             ax=axes,
             stat='percent',
             label='G|ERG EPT80: Mean: {:.2f}, Std. Dev: {:.2f}, Median: {:.2f}'.format(np.nanmean(onsetsonlysize),np.nanstd(onsetsonlysize),np.nanmedian(onsetsonlysize))
             )
sns.histplot(onsetextrasize,
             ax=axes,
             stat='percent',
             label='G|ER... EPT80: Mean: {:.2f}, Std. Dev: {:.2f}, Median: {:.2f}'.format(np.nanmean(onsetextrasize),np.nanstd(onsetextrasize),np.nanmedian(onsetextrasize))
             )
axes.legend(loc='center right')
axes.set_xlabel('Substorm size (nT)')
axes.set_ylabel('Probability (%)')
axes.set_xlim(0,2000)
plt.show()


# %%
