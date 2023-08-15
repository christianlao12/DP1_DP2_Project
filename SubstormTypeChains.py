# %% Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.dates import DateFormatter
from collections import Counter
import seaborn as sns

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)
colors = sns.color_palette("colorblind",as_cmap=True)
# %% Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophiedf['Date_UTC'] = pd.to_datetime(sophiedf['Date_UTC'])
sophiedf = sophiedf[sophiedf['Date_UTC'].between('1996','2022')].reset_index(drop=True)
sophiedf['Duration'] = np.append(np.diff(sophiedf['Date_UTC']), np.array([pd.to_timedelta(0,'h')],dtype=np.timedelta64))
sophiedf['Delbay'] = pd.to_numeric(sophiedf['Delbay'],errors='coerce')

# Isolated Onsets
iso_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(1,len(sophiedf['Date_UTC'])-2):
    if (sophiedf.iloc[i-1]['Phase'] == 1) and (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i+1]['Phase'] == 3) and (sophiedf.iloc[i+2]['Phase'] == 1):
        iso_arr[i] = 1 # GERG
        continue
    else:
        iso_arr[i] = 0
        continue

sophiedf['Isolated'] = iso_arr

# Compound Onsets
expan_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(len(sophiedf['Date_UTC'])-1):
    if (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i+1]['Phase'] == 3):
        expan_arr[i] = 1
        continue
    else:
        expan_arr[i] = 0
        continue

comp_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
comp_arr[np.setdiff1d(np.where(expan_arr),np.where(iso_arr))] = 1
sophiedf['Compound'] = comp_arr

# Excluding onsets after a convection interval
newflag_arr = sophiedf['Flag'].to_numpy().copy()
for i in range(1,len(sophiedf['Flag'])):
    if newflag_arr[i] == 1 or (newflag_arr[i-1] == 1 and sophiedf.iloc[i]['Phase'] != 1):
        newflag_arr[i] = 1
        continue
    else:
        newflag_arr[i] = 0
        continue
sophiedf['New Flag'] = newflag_arr

#%% Only Expansion Phases
expansiondf = sophiedf.iloc[np.where(sophiedf['Phase']==2)].reset_index(drop=True)
expansiondf.drop(index=0,inplace=True)
expansiondf.reset_index(drop=True,inplace=True)

# # Loading in SuperMAG Data
# supermagdatadf = pd.read_csv('Data/SuperMAGData.csv')
# supermagdatadf['Date_UTC'] = pd.to_datetime(supermagdatadf['Date_UTC'])
# supermagdatadf[supermagdatadf['Date_UTC'].between('1995','2022')]
# supermagdatadf['SML'].replace(999999, np.nan, inplace=True)
# supermagdatadf['SMU'].replace(999999, np.nan, inplace=True)

# # Loading in OMNI Data
# omnidf = pd.read_csv('Data/OMNIData.csv')
# omnidf['Date_UTC'] = pd.to_datetime(omnidf['Date_UTC'])
# omnidf = omnidf[omnidf['Date_UTC'].between('1995','2022')]

# %% Calculating chains

# Isolated Chains
isolated = np.intersect1d(np.where(expansiondf['Isolated']==1),np.where(expansiondf['New Flag']==0))
array_iso = np.zeros(len(expansiondf))
iso_onsets = expansiondf.iloc[isolated]

array_iso[isolated] = True
first_iso = np.where(np.diff(array_iso)==1)[0] + 1
last_iso = np.where(np.diff(array_iso)==-1)[0]
iso1stdf = expansiondf.iloc[first_iso]
isolastdf = expansiondf.iloc[last_iso]
isochainlengths = (isolastdf.index.to_numpy() - iso1stdf.index.to_numpy()) + 1

# Compound Chains
compound = np.intersect1d(np.where(expansiondf['Compound']==1),np.where(expansiondf['New Flag']==0))
array_comp = np.zeros(len(expansiondf))
comp_onsets = expansiondf.iloc[compound]

array_comp[compound] = True
first_comp = np.where(np.diff(array_comp)==1)[0] + 1
first_comp = np.insert(first_comp,0,0)
last_comp = np.where(np.diff(array_comp)==-1)[0]
last_comp = np.append(last_comp,len(expansiondf)-1)
comp1stdf = expansiondf.iloc[first_comp]
complastdf = expansiondf.iloc[last_comp]
compchainlengths = (complastdf.index.to_numpy() - comp1stdf.index.to_numpy()) + 1

# %% Finding longest Isolated chain

index_max_iso = np.where(isochainlengths==np.max(isochainlengths))[0]
expansiondf.loc[first_iso[index_max_iso]:last_iso[index_max_iso]]
# iso1stdf.iloc[7537]
# isolastdf.iloc[7537]

# %% Finding longest Compound chain

index_max_comp = np.where(compchainlengths==np.max(compchainlengths))[0]
expansiondf.loc[first_comp[index_max_comp[1]]:+last_comp[index_max_comp[1]]]
# comp1stdf.iloc[0]
# complastdf.iloc[0]


#%% Chain counts and densities
isochain_len, isochain_count = np.unique(isochainlengths, return_counts=True)
isochain_dens = isochain_count/np.sum(isochain_count)

isochain_count_err = 3 * np.sqrt(isochain_count)
isochain_dens_err = isochain_count_err/np.sum(isochain_count)

isochain_ratio = np.array([isochain_dens[i]/isochain_dens[i-1] for i in range(1,len(isochain_dens))])
isochain_ratio_err = np.array([np.sqrt((isochain_dens_err[i]/isochain_dens[i])**2 + (isochain_dens_err[i-1]/isochain_dens[i-1])**2) for i in range(1,len(isochain_dens))])

compchain_len, compchain_count = np.unique(compchainlengths, return_counts=True)
compchain_dens = compchain_count/np.sum(compchain_count)

compchain_count_err = 3 * np.sqrt(compchain_count)
compchain_dens_err = compchain_count_err/np.sum(compchain_count)

compchain_ratio = np.array([compchain_dens[i]/compchain_dens[i-1] for i in range(1,len(compchain_dens))])
compchain_ratio_err = np.array([np.sqrt((compchain_dens_err[i]/compchain_dens[i])**2 + (compchain_dens_err[i-1]/compchain_dens[i-1])**2) for i in range(1,len(compchain_dens))])

# %% Plot chain lengths and occurence distribution

fig, (ax, ax1) = plt.subplots(2,1, dpi=300 ,sharex=True, sharey=True)

# Isolated
label = 'No. of expansion onsets: {}, Longest Chain: {}'.format(len(iso_onsets), np.max(isochainlengths))

ax.bar(isochain_len,isochain_count,label=label)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Counts')
ax.legend(bbox_to_anchor=(1, .5), loc='center left')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'No. of expansion onsets: {}, Longest Chain: {}'.format(len(comp_onsets), np.max(compchainlengths))

ax1.bar(compchain_len,compchain_count,label=label)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Counts')
ax1.legend(bbox_to_anchor=(1, .5), loc='center left')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))


fig.tight_layout(pad=1)

# %% Plot chain lengths and occurence density

fig, (ax, ax1) = plt.subplots(2,1, dpi=300,sharex=True, sharey=True)

# Isolated
label = 'No. of expansion onsets: {}, Longest Chain: {}'.format(len(iso_onsets), np.max(isochainlengths))

ax.bar(isochain_len,isochain_dens,label=label)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Density')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'No. of expansion onsets: {}, Longest Chain: {}'.format(len(comp_onsets), np.max(compchainlengths))

ax1.bar(compchain_len,compchain_dens,label=label)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))


fig.tight_layout(pad=1)

# %% Plot Isolated chain length transition probability

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(isochain_len,isochain_dens,yerr=isochain_dens_err,label='Number of onsets: {}, max chain length: {}'.format(len(iso_onsets),np.max(isochainlengths)))
ax.set_title('Isolated Chains')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)

ax1.scatter(isochain_len[1:], isochain_ratio)
ax1.errorbar(isochain_len[1:], isochain_ratio, yerr=isochain_ratio_err, fmt='none', ecolor='r', elinewidth=1, capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,np.max(isochain_ratio)+0.1)
ax1.fill_between(isochain_len[1:], isochain_ratio[0]-isochain_ratio_err[0], isochain_ratio[0]+isochain_ratio_err[0], alpha=0.3)
ax1.hlines(0.5,0,np.max(isochainlengths),linestyles='dashed',colors='k')


fig.tight_layout(pad=1)


# %% Analysis of erer chain lengths
fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(compchain_len,compchain_dens,yerr=compchain_dens_err,label='Number of onsets: {}, max chain length: {}'.format(len(comp_onsets),np.max(compchainlengths)))
ax.set_title('Compound Chains')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)

ax1.scatter(compchain_len[1:], compchain_ratio)
ax1.errorbar(compchain_len[1:], compchain_ratio, yerr=compchain_ratio_err, fmt='none', ecolor='r', elinewidth=1, capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,np.max(compchain_ratio)+0.1)
ax1.fill_between(compchain_len[1:], compchain_ratio[1]-compchain_ratio_err[1], compchain_ratio[1]+compchain_ratio_err[1], alpha=0.3)
ax1.hlines(0.5,0,np.max(compchainlengths),linestyles='dashed',colors='k')


fig.tight_layout(pad=1)


# %%
