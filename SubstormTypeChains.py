# %% Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from matplotlib.dates import DateFormatter
from collections import Counter
import seaborn as sns

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)
colormap = sns.color_palette("colorblind",as_cmap=True)

# %% Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophiedf['Date_UTC'] = pd.to_datetime(sophiedf['Date_UTC'])
sophiedf = sophiedf[sophiedf['Date_UTC'].between('1996','2022')].reset_index(drop=True)
sophiedf['Delbay'] = pd.to_numeric(sophiedf['Delbay'],errors='coerce')
sophiedf = sophiedf.loc[2:].reset_index(drop=True)

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
exclude_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(2, len(sophiedf['Date_UTC'])-1):
    if sophiedf.iloc[i]['Phase'] == 2 and sophiedf.iloc[i+1]['Phase'] == 1:
        exclude_arr[i] = 1
        continue
    if sophiedf.iloc[i]['Phase'] == 2 and sophiedf.iloc[i-2]['Phase'] == 1:
        exclude_arr[i] = 1
    else:
        exclude_arr[i] = 0
        continue

comp_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
comp_arr[np.setdiff1d(np.where(sophiedf['Phase']==2),np.where(iso_arr==1))] = 1
comp_arr[np.where(exclude_arr==1)] = 0
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
sophiedf['NewFlag'] = newflag_arr

# Finding last onset of compound chain that are ended by a convection interval
compend_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(len(sophiedf['Date_UTC'])-2):
    if (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i]['NewFlag'] == 0) and (sophiedf.iloc[i+2]['NewFlag'] == 1):
        compend_arr[i] = 1
        continue
    else:
        compend_arr[i] = 0
        continue
sophiedf['OnsetBeforeConvection'] = compend_arr

#%% Loading in OMNI Data
# omnidf = pd.read_csv('Data/OMNIData.csv')
# omnidf['Date_UTC'] = pd.to_datetime(omnidf['Date_UTC'])
# omnidf = omnidf[omnidf['Date_UTC'].between('1995','2022')]

# %% Loading in SME Data
# smedf = pd.read_csv('Data/SMEdata.txt')
# smedf['Date_UTC'] = pd.to_datetime(smedf['Date_UTC'])
# smedf[smedf['Date_UTC'].between('1995','2022')]
# smedf['SML'].replace(999999, np.nan, inplace=True)
# smedf['SMU'].replace(999999, np.nan, inplace=True)

# %% Loading in SMR Data
# smrdf = pd.read_csv('Data/SMRdata.txt')
# smrdf['Date_UTC'] = pd.to_datetime(smrdf['Date_UTC'])
# smrdf['SMR'].replace(999999, np.nan, inplace=True)
# smrdf['SMR00'].replace(999999, np.nan, inplace=True)
# smrdf['SMR06'].replace(999999, np.nan, inplace=True)
# smrdf['SMR12'].replace(999999, np.nan, inplace=True)
# smrdf['SMR18'].replace(999999, np.nan, inplace=True)

#%% Only Expansion Phases
expansiondf = sophiedf.iloc[np.where(sophiedf['Phase']==2)].reset_index(drop=True)
expansiondf.drop(index=0,inplace=True)
expansiondf.reset_index(drop=True,inplace=True)

# %% Calculating chains

# Isolated Chains
isolated = np.intersect1d(np.where(expansiondf['Isolated']==1),np.where(expansiondf['NewFlag']==0))
array_iso = np.zeros(len(expansiondf))
iso_onsets = expansiondf.iloc[isolated]

array_iso[isolated] = True
first_iso = np.where(np.diff(array_iso)==1)[0] + 1
last_iso = np.where(np.diff(array_iso)==-1)[0]
iso1stdf = expansiondf.iloc[first_iso]
isolastdf = expansiondf.iloc[last_iso]
isochains = (isolastdf.index.to_numpy() - iso1stdf.index.to_numpy()) + 1

# Compound Chains
compound = np.intersect1d(np.where(expansiondf['Compound']==1),np.where(expansiondf['NewFlag']==0))
array_comp = np.zeros(len(expansiondf))
comp_onsets = expansiondf.iloc[compound]

array_comp[compound] = True
first_comp = np.where(np.diff(array_comp)==1)[0] + 1
first_comp = np.insert(first_comp,0,0)
last_comp = np.where(np.diff(array_comp)==-1)[0]
last_comp = np.append(last_comp,len(expansiondf)-1)
comp1stdf = expansiondf.iloc[first_comp]
complastdf = expansiondf.iloc[last_comp]
compchains = (complastdf.index.to_numpy() - comp1stdf.index.to_numpy()) + 1
compchains_growth = compchains[np.where(complastdf['OnsetBeforeConvection']==0)[0]] 
compchains_convection = compchains[np.where(complastdf['OnsetBeforeConvection']==1)[0]]

#%% Chain counts and densities

# Isolated
isochain_len, isochain_count = np.unique(isochains, return_counts=True)
isochain_dens = isochain_count/np.sum(isochain_count)

isochain_count_err = 3 * np.sqrt(isochain_count)
isochain_dens_err = isochain_count_err/np.sum(isochain_count)

isochain_ratio = np.array([isochain_dens[i]/isochain_dens[i-1] for i in range(1,len(isochain_dens))])
isochain_ratio_err = np.array([np.sqrt((isochain_dens_err[i]/isochain_dens[i])**2 + (isochain_dens_err[i-1]/isochain_dens[i-1])**2) for i in range(1,len(isochain_dens))])

# Compound
compchain_len, compchain_count = np.unique(compchains, return_counts=True)
compchain_dens = compchain_count/np.sum(compchain_count)

compchain_count_err = 3 * np.sqrt(compchain_count)
compchain_dens_err = compchain_count_err/np.sum(compchain_count)

compchain_ratio = np.array([compchain_dens[i]/compchain_dens[i-1] for i in range(1,len(compchain_dens))])
compchain_ratio_err = np.array([np.sqrt((compchain_dens_err[i]/compchain_dens[i])**2 + (compchain_dens_err[i-1]/compchain_dens[i-1])**2) for i in range(1,len(compchain_dens))])

# Compound ending in Growth
compchain_len_growth, compchain_count_growth = np.unique(compchains_growth, return_counts=True)
compchain_dens_growth = compchain_count_growth/np.sum(compchain_count_growth)

compchain_count_err_growth = 3 * np.sqrt(compchain_count_growth)
compchain_dens_err_growth = compchain_count_err_growth/np.sum(compchain_count_growth)

compchain_ratio_growth = np.array([compchain_dens_growth[i]/compchain_dens_growth[i-1] for i in range(1,len(compchain_dens_growth))])
compchain_ratio_err_growth = np.array([np.sqrt((compchain_dens_err_growth[i]/compchain_dens_growth[i])**2 + (compchain_dens_err_growth[i-1]/compchain_dens_growth[i-1])**2) for i in range(1,len(compchain_dens_growth))])

# Compound ending in Convection
compchain_len_convection, compchain_count_convection = np.unique(compchains_convection, return_counts=True)
compchain_dens_convection = compchain_count_convection/np.sum(compchain_count_convection)

compchain_count_err_convection = 3 * np.sqrt(compchain_count_convection)
compchain_dens_err_convection = compchain_count_err_convection/np.sum(compchain_count_convection)

compchain_ratio_convection = np.array([compchain_dens_convection[i]/compchain_dens_convection[i-1] for i in range(1,len(compchain_dens_convection))])
compchain_ratio_err_convection = np.array([np.sqrt((compchain_dens_err_convection[i]/compchain_dens_convection[i])**2 + (compchain_dens_err_convection[i-1]/compchain_dens_convection[i-1])**2) for i in range(1,len(compchain_dens_convection))])


# %% Plot chain lengths and occurence counts

fig, (ax, ax1) = plt.subplots(2,1, dpi=300 ,sharex=True, sharey=True)

# Isolated
label = 'No. of Chains: {}, Longest Chain: {}'.format(len(isochains), np.max(isochains))

ax.bar(isochain_len,isochain_count,label=label)
ax.errorbar(isochain_len,isochain_count,yerr=isochain_count_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Counts')
ax.legend(bbox_to_anchor=(1, .5), loc='center left')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'No. of Chains: {}, Longest Chain: {}'.format(len(compchains), np.max(compchains))

ax1.bar(compchain_len,compchain_count,label=label)
ax1.errorbar(compchain_len,compchain_count,yerr=compchain_count_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Counts')
ax1.legend(bbox_to_anchor=(1, .5), loc='center left')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))


fig.tight_layout(pad=1)

# %% Plot chain lengths and occurence probability

fig, (ax, ax1) = plt.subplots(2,1, dpi=300,sharex=True, sharey=True)

# Isolated
label = 'No. of chains: {}, Longest Chain: {}'.format(len(isochains), np.max(isochains))

ax.bar(isochain_len,isochain_dens,label=label)
ax.errorbar(isochain_len,isochain_dens,yerr=isochain_dens_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Density')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'No. of chains: {}, Longest Chain: {}'.format(len(compchains), np.max(compchains))

ax1.bar(compchain_len,compchain_dens,label=label)
ax1.errorbar(compchain_len,compchain_dens,yerr=compchain_dens_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

fig.tight_layout(pad=1)

# %% Plot Compound chain lengths and occurence counts (All, Growth, Convection)
barplotdf = pd.DataFrame({'Length':compchain_len[compchain_len<17],
                            'All Count':compchain_count[compchain_len<17],
                            'Growth Count':compchain_count_growth[compchain_len_growth<17],
                            'Convection Count':compchain_count_convection[compchain_len_convection<17]})
barplotdf.set_index('Length',inplace=True)

fig, ax = plt.subplots(dpi=300)

barplotdf.plot(ax=ax,
                kind='bar',
                xlabel='Length of repeating pattern',
                ylabel='Counts',
                rot=0,
                legend=True,
                )

# %%
index = np.where(compchains==1)[0]
nonconvection = np.where(complastdf['OnsetBeforeConvection']==0)


# %% Plot Isolated chain length transition probability

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(isochain_len,isochain_dens,label='Number of onsets: {}, max chain length: {}'.format(len(iso_onsets),np.max(isochains)))
ax.errorbar(isochain_len,isochain_dens,yerr=isochain_dens_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
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
ax1.hlines(0.5,0,np.max(isochains),linestyles='dashed',color='k')
ax1.set_xlim(0,18.5)

fig.tight_layout(pad=1)

# %% Plot Compound chain length transition probability
fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(compchain_len,compchain_dens,label='Number of onsets: {}, max chain length: {}'.format(len(comp_onsets),np.max(compchains)))
ax.errorbar(compchain_len,compchain_dens,yerr=compchain_dens_err,fmt='none',ecolor='k',elinewidth=1,capsize=2)
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
ax1.hlines(0.5,0,np.max(compchains),linestyles='dashed',color='k')
ax1.set_xlim(0,18.5)

fig.tight_layout(pad=1)

# %% Monthly occurence chains of length >= 2
isochain_greq_2 = np.where(isochains>=2)[0]
isochain_greq_2_array = np.zeros(len(isochain_greq_2),dtype=int)

for i in range(len(isochain_greq_2)):
    isochain_greq_2_array[i] = iso1stdf.iloc[isochain_greq_2[i]]['Date_UTC'].month

isochain_greq_2_months, isochain_greq_2_months_count = np.unique(isochain_greq_2_array, return_counts=True)
isochain_greq_2_months_dens = isochain_greq_2_months_count/np.sum(isochain_greq_2_months_count)

compchain_greq_2 = np.where(compchains>=2)[0]
compchain_greq_2_array = np.zeros(len(compchain_greq_2),dtype=int)

for i in range(len(compchain_greq_2)):
    compchain_greq_2_array[i] = comp1stdf.iloc[compchain_greq_2[i]]['Date_UTC'].month

compchain_greq_2_months, compchain_greq_2_months_count = np.unique(compchain_greq_2_array, return_counts=True)
compchain_greq_2_months_dens = compchain_greq_2_months_count/np.sum(compchain_greq_2_months_count)

fig, ax = plt.subplots(2,1,dpi=300,sharex=True,sharey=True)

ax[0].bar(isochain_greq_2_months,isochain_greq_2_months_dens,label='Number of onsets: {}'.format(np.sum(isochains[isochain_greq_2])))
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Monthly occurence of First Onset of Isolated chains (Length >= 2)')
ax[0].legend(loc='best')
ax[0].xaxis.set_tick_params(labelbottom=True)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

ax[1].bar(compchain_greq_2_months,compchain_greq_2_months_dens,color=colormap[1], label='Number of onsets: {}'.format(np.sum(compchains[compchain_greq_2])))
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('Monthly occurence of First Onset of Compound chains (Length >= 2)')
ax[1].legend(loc='best')

fig.tight_layout(pad=1)

# %% Monthly occurence chains of length >= 2 in Mar 2003 to Feb 2004

isochain_greq_2_0304_array = []
isochain_greq_2_0304 = []

for i in range(len(isochain_greq_2)):
    if pd.to_datetime("2003-03") <= iso1stdf.iloc[isochain_greq_2[i]]['Date_UTC'] <= pd.to_datetime("2004-03"):
        isochain_greq_2_0304_array.append(iso1stdf.iloc[isochain_greq_2[i]]['Date_UTC'].month)
        isochain_greq_2_0304.append(isochain_greq_2[i])

isochain_greq_2_0304_months, isochain_greq_2_0304_months_count = np.unique(isochain_greq_2_0304_array, return_counts=True)
isochain_greq_2_0304_months_dens = isochain_greq_2_0304_months_count/np.sum(isochain_greq_2_0304_months_count)

compchain_greq_2_0304_array = []
compchain_greq_2_0304 = []

for i in range(len(compchain_greq_2)):
    if pd.to_datetime("2003-03") <= comp1stdf.iloc[compchain_greq_2[i]]['Date_UTC'] <= pd.to_datetime("2004-03"):
        compchain_greq_2_0304_array.append(comp1stdf.iloc[compchain_greq_2[i]]['Date_UTC'].month)
        compchain_greq_2_0304.append(compchain_greq_2[i])

compchain_greq_2_0304_months, compchain_greq_2_0304_months_count = np.unique(compchain_greq_2_0304_array, return_counts=True)
compchain_greq_2_0304_months_dens = compchain_greq_2_0304_months_count/np.sum(compchain_greq_2_0304_months_count)

fig, ax = plt.subplots(2,1,dpi=300,sharex=True,sharey=True)

ax[0].bar(
        isochain_greq_2_0304_months,
        isochain_greq_2_0304_months_dens,
        label='Number of onsets: {}'.format(np.sum(isochains[isochain_greq_2_0304]))
        )
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Monthly occurence of First Onset of Isolated chains (Length >= 2)')
ax[0].legend(loc='best')
ax[0].xaxis.set_tick_params(labelbottom=True)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

ax[1].bar(
        compchain_greq_2_0304_months,
        compchain_greq_2_0304_months_dens,color=colormap[1],
        label='Number of onsets: {}'.format(np.sum(compchains[compchain_greq_2_0304]))
        )
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('Monthly occurence of First Onset of Compound chains (Length >= 2)')
ax[1].legend(loc='best')

fig.suptitle('Mar 2003 to Feb 2004 (inclusive)')
fig.tight_layout(pad=1)
# %% Isolated chain length vs IMF Bz

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(-15,15.5,1)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(isochain_len[isochain_len<19])):
    chain_len_index = np.where(isochains==isochain_len[i])[0]

    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].values)
    
    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)


values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Bz (nT)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %% Compound chain length vs IMF Bz

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(-15,15.5,1)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(compchain_len[compchain_len<19])):
    chain_len_index = np.where(compchains==compchain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].values)
    
    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)


values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Bz (nT)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)
# %% Isolated chain length vs Vsw

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(200,800,25)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(isochain_len[isochain_len<19])):
    chain_len_index = np.where(isochains==isochain_len[i])[0]

    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].values)
    
    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)


values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Flow Speed (km/s)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

fig.tight_layout(pad=1)

# %% Compound chain length vs Vsw

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(200,800,25)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(compchain_len[compchain_len<19])):
    chain_len_index = np.where(compchains==compchain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].values)
    
    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)


values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Flow Speed (km/s)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

fig.tight_layout(pad=1)
# %% Isolated chain length vs Nsw

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(0,25.5,0.5)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(isochain_len[isochain_len<19])):
    chain_len_index = np.where(isochains==isochain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')

        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['proton_density'].values)

    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)

# Plotting
values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Density (cm$^{-3}$)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.set_ylim(bottom=0)

fig.tight_layout(pad=1)

# %% Compound chain length vs Nsw

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(0,25.5,0.5)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(compchain_len[compchain_len<19])):
    chain_len_index = np.where(compchains==compchain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')

        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['proton_density'].values)

    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)

# Plotting
values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compoound Chain Length")
ax.set_ylabel("Density (cm$^{-3}$)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.set_ylim(bottom=0)

fig.tight_layout(pad=1)

# %% Isolated chain length vs SMR

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(-100,51,2.5)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(isochain_len[isochain_len<19])):
    chain_len_index = np.where(isochains==isochain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')

        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].values)

    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)

# Plotting
values = np.array(chain_hist)
fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3,)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("SMR (nT)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

fig.tight_layout(pad=1)

# %% Compound chain length vs SMR

x_edges = np.arange(1,20) - 0.5
y_edges = np.arange(-100,51,2.5)

chain_vals = []
chain_hist = []
chain_mean = []
chain_std = []

for i in range(len(compchain_len[compchain_len<19])):
    chain_len_index = np.where(compchains==compchain_len[i])[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(1,'h')

        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].values)

    chain_vals_loop = np.concatenate(chain_vals_loop).ravel()
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_mean_loop = np.nanmean(chain_vals_loop)
    chain_std_loop = np.nanstd(chain_vals_loop)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_mean.append(chain_mean_loop)
    chain_std.append(chain_std_loop)

# Plotting
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

X,Y = np.meshgrid(x_edges,y_edges)
plot = ax.pcolormesh(X, Y, values.T, cmap='viridis', vmax=0.3)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("SMR (nT)")
ax.scatter(np.arange(1,19),chain_mean,c=colormap[3],marker='x',label='Mean')
ax.errorbar(np.arange(1,19),chain_mean,yerr=chain_std,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='St. Dev')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

fig.tight_layout(pad=1)
