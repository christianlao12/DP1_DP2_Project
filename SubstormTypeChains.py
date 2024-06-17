# %%
# Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from matplotlib.dates import DateFormatter
from collections import Counter
from scipy import stats
import seaborn as sns

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)
colormap = sns.color_palette("colorblind",as_cmap=True)

# Function to calculate chi squared test
def chi_squared_test(measured,model,uncertainty):
    return np.sum(np.nan_to_num((np.array(measured)-np.array(model))**2/np.array(uncertainty)**2))


# %% 
# Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT80_1990-2022.csv", low_memory=False)
sophiedf['Date_UTC'] = pd.to_datetime(sophiedf['Date_UTC'])
sophiedf['Duration'] = np.append(np.diff(sophiedf["Date_UTC"].to_numpy()),0)
sophiedf = sophiedf[sophiedf['Date_UTC'].between('1998','2022')].reset_index(drop=True)
sophiedf['Delbay'] = pd.to_numeric(sophiedf['Delbay'],errors='coerce')
sophiedf = sophiedf.loc[2:].reset_index(drop=True)

# %%
# SOPHIE Phases

# Isolated Onsets
iso_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(1,len(sophiedf['Date_UTC'])-2):
    if (sophiedf.iloc[i-1]['Phase'] == 1) and (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i+1]['Phase'] == 3) and (sophiedf.iloc[i+2]['Phase'] == 1):
        iso_arr[i] = 1 # GERG

sophiedf['Isolated Onset'] = iso_arr

# Compound Onsets
# Excluding expansion phases directly before growth phases
expansionbeforegrowth_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(len(sophiedf['Date_UTC'])-1):
    if (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i+1]['Phase'] == 1):
        expansionbeforegrowth_arr[i] = 1

for i in reversed(range(len(sophiedf['Date_UTC'])-2)):
    if (sophiedf.iloc[i]['Phase']==2) and (expansionbeforegrowth_arr[i+2] == 1):
        expansionbeforegrowth_arr[i] = 1

# Excluding expansion phases directly after recovery phases that follow a growth phase
expansionafterGR_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
for i in range(2,len(sophiedf['Date_UTC'])):
    if (sophiedf.iloc[i]['Phase'] == 2) and (sophiedf.iloc[i-1]['Phase'] == 3) and (sophiedf.iloc[i-2]['Phase'] == 1):
        expansionafterGR_arr[i] = 1

for i in range(2,len(sophiedf['Date_UTC'])):
    if (sophiedf.iloc[i]['Phase'] == 2) and (expansionafterGR_arr[i-2] == 1):
        expansionafterGR_arr[i] = 1

compound_arr = np.zeros(len(sophiedf['Date_UTC']),dtype=int)
compound_arr[np.setdiff1d(np.where(sophiedf['Phase']==2),np.where(iso_arr==1))] = 1
compound_arr[np.where(expansionbeforegrowth_arr==1)] = 0
compound_arr[np.where(expansionafterGR_arr==1)] = 0
sophiedf['Compound Onset'] = compound_arr

# Excluding onsets after a convection interval
newflag_arr = sophiedf['Flag'].to_numpy().copy()
for i in range(1,len(sophiedf['Flag'])):
    if newflag_arr[i] == 1 or (newflag_arr[i-1] == 1 and sophiedf.iloc[i]['Phase'] != 1):
        newflag_arr[i] = 1

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

#%% 
# Calculating chains

# Only Expansion Phases
expansiondf = sophiedf.iloc[np.where(sophiedf['Phase']==2)].reset_index(drop=True)

# Only Convection Expansions
convec_expansiondf = expansiondf.iloc[np.where(expansiondf['Flag']==1)]

# Isolated Chains
isolated = np.intersect1d(np.where(expansiondf['Isolated Onset']==1),np.where(expansiondf['NewFlag']==0))
iso_onsets = expansiondf.iloc[isolated]

array_iso = np.zeros(len(expansiondf))
array_iso[isolated] = True
first_iso = np.where(np.diff(array_iso)==1)[0] + 1
last_iso = np.where(np.diff(array_iso)==-1)[0]
iso1stdf = expansiondf.iloc[first_iso]
isolastdf = expansiondf.iloc[last_iso]
isochains = (isolastdf.index.to_numpy() - iso1stdf.index.to_numpy()) + 1

# Compound Chains
compound = np.intersect1d(np.where(expansiondf['Compound Onset']==1),np.where(expansiondf['NewFlag']==0))
comp_onsets = expansiondf.iloc[compound]

array_comp = np.zeros(len(expansiondf))
array_comp[compound] = True
first_comp = np.where(np.diff(array_comp)==1)[0] + 1
last_comp = np.where(np.diff(array_comp)==-1)[0]
last_comp = np.append(last_comp,len(expansiondf)-1)
comp1stdf = expansiondf.iloc[first_comp]
subseq_comp = expansiondf.iloc[np.setdiff1d(compound,first_comp)]
complastdf = expansiondf.iloc[last_comp]
compchains = (complastdf.index.to_numpy() - comp1stdf.index.to_numpy()) + 1
compchains_growth = compchains[np.where(complastdf['OnsetBeforeConvection']==0)[0]] 
compchains_convec = compchains[np.where(complastdf['OnsetBeforeConvection']==1)[0]]

comp_onsets_growth = pd.DataFrame()
comp_onsets_growth_index = list(zip(first_comp[np.where(complastdf['OnsetBeforeConvection']==0)[0]],last_comp[np.where(complastdf['OnsetBeforeConvection']==0)[0]] + 1))

for i in range(len(comp_onsets_growth_index)):
    comp_onsets_growth = pd.concat([comp_onsets_growth,expansiondf.iloc[comp_onsets_growth_index[i][0]:comp_onsets_growth_index[i][1]]])

comp_onsets_convec = pd.DataFrame()
comp_onsets_convec_index = list(zip(first_comp[np.where(complastdf['OnsetBeforeConvection']==1)[0]],last_comp[np.where(complastdf['OnsetBeforeConvection']==1)[0]] + 1))

for i in range(len(comp_onsets_convec_index)):
    comp_onsets_convec = pd.concat([comp_onsets_convec,expansiondf.iloc[comp_onsets_convec_index[i][0]:comp_onsets_convec_index[i][1]]])

after_convec = np.intersect1d(np.where(expansiondf['Phase']==2),np.setdiff1d(np.where(expansiondf['NewFlag']==1)[0],np.where(expansiondf['Flag']==1)[0]))
onsets_after_convec = expansiondf.iloc[after_convec]

geg = np.setdiff1d(expansiondf.index.to_numpy(),np.union1d(np.union1d(np.union1d(isolated,compound),np.where(expansiondf['Flag']==1)),after_convec))
gegdf = expansiondf.iloc[geg]

# %%
# Substorm size distributions

# Isolated
isolated_onsets_size = np.abs(iso_onsets['Delbay'].to_numpy())
log_isolated_onsets_size = np.log10(isolated_onsets_size)

# Compound
compound_onsets_size = np.abs(comp_onsets['Delbay'].to_numpy())
log_compound_onsets_size = np.log10(compound_onsets_size)

# Convection expansions
convec_onsets_size = np.abs(comp_onsets_convec['Delbay'].to_numpy())
log_convec_onsets_size = np.log10(convec_onsets_size)

# Plotting
fig, ax = plt.subplots(dpi=300)

ax.hist(isolated_onsets_size,color=colormap[1],bins=np.geomspace(10,2100,100),label='Isolated: No. of onsets: {}'.format(len(iso_onsets)),histtype='step',density=True,stacked=True)
ax.hist(compound_onsets_size,bins=np.geomspace(10,2100,100),color=colormap[2], label='Compound: No. of onsets: {}'.format(len(comp_onsets)),histtype='step',density=True,stacked=True)
ax.hist(convec_onsets_size,bins=np.geomspace(10,2100,100),color=colormap[3], label='Convection Expansion Phase: No. of onsets: {}'.format(len(comp_onsets_convec)),histtype='step',density=True,stacked=True)
ax.set_xlabel('Substorm Size (nT)')
ax.set_ylabel('Probability')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend(loc='best')

fig.tight_layout(pad=1)

# KS Test
print('KS Test: Isolated vs Compound: {}'.format(stats.ks_2samp(isolated_onsets_size,compound_onsets_size)))
print('KS Test: Isolated vs Convection Expansion: {}'.format(stats.ks_2samp(isolated_onsets_size,convec_onsets_size)))
print('KS Test: Compound vs Convection Expansion: {}'.format(stats.ks_2samp(compound_onsets_size,convec_onsets_size)))

# Normality Test
print('Normality Test: Isolated: {}'.format(stats.normaltest(log_isolated_onsets_size)))
print('Normality Test: Compound: {}'.format(stats.normaltest(log_compound_onsets_size)))
print('Normality Test: Convection Expansion: {}'.format(stats.normaltest(log_convec_onsets_size)))

# %%
# Substorm MLT distributions

# All onsets
onsets_mlt = sophiedf.iloc[np.where(sophiedf['Phase']==2)]['MLT'].to_numpy()

onsets_mlt_counts, onsets_mlt_bins = np.histogram(onsets_mlt,bins=np.arange(0,25))
onsets_mlt_bins = onsets_mlt_bins[:-1]
onsets_mlt_counts = [*onsets_mlt_counts[12:],*onsets_mlt_counts[:12]]
onsets_mlt_counts_err = 2 * np.sqrt(onsets_mlt_counts)
onsets_mlt_dens = onsets_mlt_counts/np.sum(onsets_mlt_counts)
onsets_mlt_dens_err = onsets_mlt_counts_err/np.sum(onsets_mlt_counts)

# Isolated
isolated_onsets_mlt = iso_onsets['MLT'].to_numpy()

iso_mlt_counts, iso_mlt_bins = np.histogram(isolated_onsets_mlt,bins=np.arange(0,25))
iso_mlt_bins = iso_mlt_bins[:-1]
iso_mlt_counts = [*iso_mlt_counts[12:],*iso_mlt_counts[:12]]
iso_mlt_counts_err = 2 * np.sqrt(iso_mlt_counts)
iso_mlt_dens = iso_mlt_counts/np.sum(iso_mlt_counts)
iso_mlt_dens_err = iso_mlt_counts_err/np.sum(iso_mlt_counts)

# Compound
compound_onsets_mlt = comp_onsets['MLT'].to_numpy()
comp_mlt_counts, comp_mlt_bins = np.histogram(compound_onsets_mlt,bins=np.arange(0,25))
comp_mlt_bins = comp_mlt_bins[:-1]
comp_mlt_counts = [*comp_mlt_counts[12:],*comp_mlt_counts[:12]]
comp_mlt_counts_err = 2 * np.sqrt(comp_mlt_counts)
comp_mlt_dens = comp_mlt_counts/np.sum(comp_mlt_counts)
comp_mlt_dens_err = comp_mlt_counts_err/np.sum(comp_mlt_counts)

# Convection expansions
convec_onsets_mlt = convec_expansiondf['MLT'].to_numpy()
convec_mlt_counts, convec_mlt_bins = np.histogram(convec_onsets_mlt,bins=np.arange(0,25))
convec_mlt_bins = convec_mlt_bins[:-1]
convec_mlt_counts = [*convec_mlt_counts[12:],*convec_mlt_counts[:12]]
convec_mlt_counts_err = 2 * np.sqrt(convec_mlt_counts)
convec_mlt_dens = convec_mlt_counts/np.sum(convec_mlt_counts)
convec_mlt_dens_err = convec_mlt_counts_err/np.sum(convec_mlt_counts)

# After convection expansions
after_convec_mlt = onsets_after_convec['MLT'].to_numpy()
after_convec_mlt_counts, after_convec_mlt_bins = np.histogram(after_convec_mlt,bins=np.arange(0,25))
after_convec_mlt_bins = after_convec_mlt_bins[:-1]
after_convec_mlt_counts = [*after_convec_mlt_counts[12:],*after_convec_mlt_counts[:12]]
after_convec_mlt_counts_err = 2 * np.sqrt(after_convec_mlt_counts)
after_convec_mlt_dens = after_convec_mlt_counts/np.sum(after_convec_mlt_counts)
after_convec_mlt_dens_err = after_convec_mlt_counts_err /np.sum(after_convec_mlt_counts)

# GEG expansions
geg_mlt = gegdf['MLT'].to_numpy()
geg_mlt_counts, geg_mlt_bins = np.histogram(geg_mlt,bins=np.arange(0,25))
geg_mlt_bins = geg_mlt_bins[:-1]
geg_mlt_counts = [*geg_mlt_counts[12:],*geg_mlt_counts[:12]]
geg_mlt_counts_err = 2 * np.sqrt(geg_mlt_counts)
geg_mlt_dens = geg_mlt_counts/np.sum(geg_mlt_counts)
geg_mlt_dens_err = geg_mlt_counts_err/np.sum(geg_mlt_counts)


# Labelling of the MLT bins
bins = list(range(12,24)) + list(range(0,12))
bins = list(map(str,bins))  

# Plotting histograms (All onsets, Isolated, Compound, Convection Expansion, After Convection Expansion, GEG Expansion)
fig, ax = plt.subplots(dpi=300)

ax.plot(onsets_mlt_counts,color=colormap[0],label='All Onsets: No. of onsets: {}'.format(len(onsets_mlt)))
ax.plot(iso_mlt_counts,color=colormap[1],label='Isolated: No. of onsets: {}'.format(len(iso_onsets)))
ax.plot(comp_mlt_counts,color=colormap[2],label='Compound: No. of onsets: {}'.format(len(comp_onsets)))
ax.plot(convec_mlt_counts,color=colormap[3],label='Convection Expansion Phase: No. of onsets: {}'.format(len(convec_expansiondf)))
ax.plot(after_convec_mlt_counts,color=colormap[4],ls='--',label='After Convection Expansion Phase: No. of onsets: {}'.format(len(onsets_after_convec)))
ax.plot(geg_mlt_counts,color=colormap[5],ls='--',label='GEG Expansion Phase: No. of onsets: {}'.format(len(gegdf)))
ax.set_xlabel('MLT')
ax.set_ylabel('Counts')
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting probability distributions (All onsets, Isolated, Compound, Convection Expansion, After Convection Expansion, GEG Expansion)

fig, ax= plt.subplots(dpi=300)

ax.plot(onsets_mlt_dens,color=colormap[0],label='All Onsets: No. of onsets: {}'.format(len(onsets_mlt)))
ax.plot(iso_mlt_dens,color=colormap[1],label='Isolated: No. of onsets: {}'.format(len(iso_onsets)))
ax.plot(comp_mlt_dens,color=colormap[2], label='Compound: No. of onsets: {}'.format(len(comp_onsets)))
ax.plot(convec_mlt_dens,color=colormap[3],label='Convection Expansion Phase: No. of onsets: {}'.format(len(convec_expansiondf)))
ax.plot(after_convec_mlt_dens,color=colormap[4],ls='--',label='After Convection Expansion Phase: No. of onsets: {}'.format(len(onsets_after_convec)))
ax.plot(geg_mlt_dens,color=colormap[5],ls='--',label='GEG Expansion Phase: No. of onsets: {}'.format(len(gegdf)))
ax.set_xlabel('MLT')
ax.set_ylabel('Probability')
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc='lower center',bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# %%
# Substorm onset UT distributions

# Isolated
isolated_onsets_ut = iso_onsets['Date_UTC'].dt.hour.to_numpy()
iso_ut_counts, iso_ut_bins = np.histogram(isolated_onsets_ut,bins=np.arange(0,25))
iso_ut_bins = iso_ut_bins[:-1]
iso_ut_dens = iso_ut_counts/np.sum(iso_ut_counts)
iso_ut_dens = [*iso_ut_dens[12:],*iso_ut_dens[:12]]
iso_ut_dens_err = 2 * np.sqrt(iso_ut_counts)/np.sum(iso_ut_counts)

# Compound
compound_onsets_ut = comp_onsets['Date_UTC'].dt.hour.to_numpy()
comp_ut_counts, comp_ut_bins = np.histogram(compound_onsets_ut,bins=np.arange(0,25))
comp_ut_bins = comp_ut_bins[:-1]
comp_ut_dens = comp_ut_counts/np.sum(comp_ut_counts)
comp_ut_dens = [*comp_ut_dens[12:],*comp_ut_dens[:12]]
comp_ut_dens_err = 2 * np.sqrt(comp_ut_counts)/np.sum(comp_ut_counts)

# Convection
convection_onsets_ut = convec_expansiondf['Date_UTC'].dt.hour.to_numpy()
convec_ut_counts, convec_ut_bins = np.histogram(convection_onsets_ut,bins=np.arange(0,25))
convec_ut_bins = convec_ut_bins[:-1]
convec_ut_dens = convec_ut_counts/np.sum(convec_ut_counts)
convec_ut_dens = [*convec_ut_dens[12:],*convec_ut_dens[:12]]
convec_ut_dens_err = 2 * np.sqrt(convec_ut_counts)/np.sum(convec_ut_counts)

# Isolated between 2 and 12 MLT
isolated_onsets_2_10_ut = iso_onsets.iloc[np.intersect1d(np.where(iso_onsets['MLT']>=2)[0],np.where(iso_onsets['MLT']<=10)[0])]['Date_UTC'].dt.hour.to_numpy()
iso_ut_2_10_counts, iso_ut_2_10_bins = np.histogram(isolated_onsets_2_10_ut,bins=np.arange(0,25))
iso_ut_2_10_bins = iso_ut_2_10_bins[:-1]
iso_ut_2_10_dens = iso_ut_2_10_counts/np.sum(iso_ut_2_10_counts)
iso_ut_2_10_dens = [*iso_ut_2_10_dens[12:],*iso_ut_2_10_dens[:12]]
iso_ut_2_10_dens_err = 2 * np.sqrt(iso_ut_2_10_counts)/np.sum(iso_ut_2_10_counts)

bins = list(range(12,24)) + list(range(0,12))
bins = list(map(str,bins))

# Plotting
fig, ax= plt.subplots(dpi=300)

ax.errorbar(x=iso_ut_bins,y=iso_ut_dens,yerr=comp_ut_dens_err,color=colormap[0],capsize=2,label='Isolated: No. of onsets: {}'.format(len(iso_onsets)))
ax.errorbar(x=comp_ut_bins,y=comp_ut_dens,yerr=comp_ut_dens_err,color=colormap[1],capsize=2,label='Compound: No. of onsets: {}'.format(len(comp_onsets)))
# ax.errorbar(x=iso_ut_2_10_bins,y=iso_ut_2_10_dens,yerr=iso_ut_2_10_dens_err,color=colormap[2],capsize=2,label=f'Isolated from 2-10 MLT: No. of onsets: {len(isolated_onsets_2_10_ut)}')
ax.errorbar(x=convec_ut_bins,y=convec_ut_dens,yerr=convec_ut_dens_err,color=colormap[3],capsize=2,label='Convection Expansion Phase: No. of onsets: {}'.format(len(convec_expansiondf)))
ax.set_xlabel('Substorm Onset UT')
ax.set_ylabel('Probability')
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc='upper left')
ax.set_ylim(0,0.1)

fig.tight_layout(pad=1)

#%% 
# Chain counts and densities calculations

# Isolated
isochain_len, isochain_count = np.unique(isochains, return_counts=True)
isochain_count_err = 2 * np.sqrt(isochain_count)

isochain_dens = isochain_count/np.sum(isochain_count)
isochain_dens_err = isochain_count_err/np.sum(isochain_count)

isochain_ratio = np.array([isochain_dens[i]/isochain_dens[i-1] for i in range(1,len(isochain_dens))])
isochain_ratio_err = np.array([np.sqrt((isochain_dens_err[i]/isochain_dens[i])**2 + (isochain_dens_err[i-1]/isochain_dens[i-1])**2) for i in range(1,len(isochain_dens))])

# Compound
compchain_len, compchain_count = np.unique(compchains, return_counts=True)
compchain_count_err = 2 * np.sqrt(compchain_count)

compchain_dens = compchain_count/np.sum(compchain_count)
compchain_dens_err = compchain_count_err/np.sum(compchain_count)

compchain_ratio = np.array([compchain_dens[i]/compchain_dens[i-1] for i in range(1,len(compchain_dens))])
compchain_ratio_err = np.array([np.sqrt((compchain_dens_err[i]/compchain_dens[i])**2 + (compchain_dens_err[i-1]/compchain_dens[i-1])**2) for i in range(1,len(compchain_dens))])

# Compound ending in Growth
compchain_len_growth, compchain_count_growth = np.unique(compchains_growth, return_counts=True)
compchain_count_err_growth = 2 * np.sqrt(compchain_count_growth)

compchain_dens_growth = compchain_count_growth/np.sum(compchain_count_growth)
compchain_dens_err_growth = compchain_count_err_growth/np.sum(compchain_count_growth)

compchain_ratio_growth = np.array([compchain_dens_growth[i]/compchain_dens_growth[i-1] for i in range(1,len(compchain_dens_growth))])
compchain_ratio_err_growth = np.array([np.sqrt((compchain_dens_err_growth[i]/compchain_dens_growth[i])**2 + (compchain_dens_err_growth[i-1]/compchain_dens_growth[i-1])**2) for i in range(1,len(compchain_dens_growth))])

# Compound ending in Convection
compchain_len_convection, compchain_count_convection = np.unique(compchains_convec, return_counts=True)
compchain_count_err_convection = 2 * np.sqrt(compchain_count_convection)

compchain_dens_convection = compchain_count_convection/np.sum(compchain_count_convection)
compchain_dens_err_convection = compchain_count_err_convection/np.sum(compchain_count_convection)

compchain_ratio_convection = np.array([compchain_dens_convection[i]/compchain_dens_convection[i-1] for i in range(1,len(compchain_dens_convection))])
compchain_ratio_err_convection = np.array([np.sqrt((compchain_dens_err_convection[i]/compchain_dens_convection[i])**2 + (compchain_dens_err_convection[i-1]/compchain_dens_convection[i-1])**2) for i in range(1,len(compchain_dens_convection))])


# %% 
# Plot chain lengths and occurence counts

fig, (ax, ax1) = plt.subplots(2,1, dpi=300 ,sharex=True, sharey=True)

# Isolated
label = 'Isolated: No. of Chains: {}, Longest Chain: {}'.format(len(isochains), np.max(isochains))
ax.bar(isochain_len,isochain_count,label=label)
ax.errorbar(isochain_len,isochain_count,yerr=isochain_count_err,fmt='none',ecolor='k',capsize=2)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Counts')
ax.legend(bbox_to_anchor=(1, .5), loc='center left')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'Compound: No. of Chains: {}, Longest Chain: {}'.format(len(compchains), np.max(compchains))
ax1.bar(compchain_len,compchain_count,color=colormap[1],label=label)
ax1.errorbar(compchain_len,compchain_count,yerr=compchain_count_err,fmt='none',ecolor='k',capsize=2)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Counts')
ax1.legend(bbox_to_anchor=(1, .5), loc='center left')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

fig.tight_layout(pad=1)

# Plot chain lengths and occurence probability

fig, (ax, ax1) = plt.subplots(2,1, dpi=300,sharex=True, sharey=True)
# Isolated
label = 'Isolated: No. of chains: {}, Longest Chain: {}'.format(len(isochains), np.max(isochains))
ax.bar(isochain_len,isochain_dens,label=label)
ax.errorbar(isochain_len,isochain_dens,yerr=isochain_dens_err,fmt='none',ecolor='k',capsize=2)
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Density')
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(labelbottom=True)

# Compound
label = 'Compound: No. of chains: {}, Longest Chain: {}'.format(len(compchains), np.max(compchains))
ax1.bar(compchain_len,compchain_dens,color=colormap[1], label=label)
ax1.errorbar(compchain_len,compchain_dens,yerr=compchain_dens_err,fmt='none',ecolor='k',capsize=2)
ax1.set_xlabel('Length of repeating pattern')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

fig.tight_layout(pad=1)

# %%
# Finding what ends an isolated chain
iso_chain_compound = []
iso_chain_convec = []
iso_chain_expansion = []
iso_chain_recovery = []


for i, __ in enumerate(isochains):
    if sophiedf.iloc[np.where(sophiedf['Date_UTC'] == isolastdf.iloc[i]['Date_UTC'])[0][0]+3]['Phase'] == 2:
        if sophiedf.iloc[np.where(sophiedf['Date_UTC'] == isolastdf.iloc[i]['Date_UTC'])[0][0]+3]['Compound Onset'] == 1:
            iso_chain_compound.append(isochains[i])
        elif sophiedf.iloc[np.where(sophiedf['Date_UTC'] == isolastdf.iloc[i]['Date_UTC'])[0][0]+3]['Flag'] == 1:
            iso_chain_convec.append(isochains[i])
        else:
            iso_chain_expansion.append(isochains[i])
    
    if sophiedf.iloc[np.where(sophiedf['Date_UTC'] == isolastdf.iloc[i]['Date_UTC'])[0][0]+3]['Phase'] == 3:
        iso_chain_recovery.append(isochains[i])


iso_chain_compound_count, iso_chain_compound_len = np.histogram(iso_chain_compound, bins=np.arange(1,12))
iso_chain_compound_count = iso_chain_compound_count[:-1]
iso_chain_compound_len = iso_chain_compound_len[:-2]
iso_chain_compound_dens = iso_chain_compound_count/np.sum(iso_chain_compound_count)

iso_chain_convec_count, iso_chain_convec_len = np.histogram(iso_chain_convec, bins=np.arange(1,12))
iso_chain_convec_count = iso_chain_convec_count[:-1]
iso_chain_convec_len = iso_chain_convec_len[:-2]
iso_chain_convec_dens = iso_chain_convec_count/np.sum(iso_chain_convec_count)

iso_chain_expansion_count, iso_chain_expansion_len = np.histogram(iso_chain_expansion, bins=np.arange(1,12))
iso_chain_expansion_count = iso_chain_expansion_count[:-1]
iso_chain_expansion_len = iso_chain_expansion_len[:-2]
iso_chain_expansion_dens = iso_chain_expansion_count/np.sum(iso_chain_expansion_count)

iso_chain_recovery_count, iso_chain_recovery_len = np.histogram(iso_chain_recovery, bins=np.arange(1,12))
iso_chain_recovery_count = iso_chain_recovery_count[:-1]
iso_chain_recovery_len = iso_chain_recovery_len[:-2]
iso_chain_recovery_dens = iso_chain_recovery_count/np.sum(iso_chain_recovery_count)

# Plotting 
isobarplotdf = pd.DataFrame({'Length':isochain_len[isochain_len<10],
                            f'All Count ({len(isochains)})':isochain_count[isochain_len<10],
                            f'Ending in Compount Count ({len(iso_chain_compound)})':iso_chain_compound_count,
                            f'Ending in Convection Count ({len(iso_chain_convec)})':iso_chain_convec_count,
                            f'Ending in Expansion Count ({len(iso_chain_expansion)})':iso_chain_expansion_count,
                            f'Ending in Recovery Count ({len(iso_chain_recovery)})':iso_chain_recovery_count})
isobarplotdf.set_index('Length',inplace=True)

fig, ax = plt.subplots(dpi=300)

isobarplotdf.plot(ax=ax,
                kind='bar',
                xlabel='Length of repeating pattern',
                ylabel='Counts',
                rot=0,
                legend=True,
                )

proportionbarplotdf = pd.DataFrame({'Length':isochain_len[isochain_len<10],
                            f'Ending in Compound Proportion':iso_chain_compound_count/isochain_count[:9],
                            f'Ending in Convection Proportion':iso_chain_convec_count/isochain_count[:9],
                            f'Ending in Expansion Proportion':iso_chain_expansion_count/isochain_count[:9],
                            f'Ending in Recovery Proportion':iso_chain_recovery_count/isochain_count[:9]})
proportionbarplotdf.set_index('Length',inplace=True)

proportionbarploterror = pd.DataFrame({'Length':isochain_len[isochain_len<10],
                                    f'Ending in Compound Proportion': 2 * np.sqrt(iso_chain_compound_count)/isochain_count[:9],
                                    f'Ending in Convection Proportion': 2 * np.sqrt(iso_chain_convec_count)/isochain_count[:9],
                                    f'Ending in Expansion Proportion': 2 * np.sqrt(iso_chain_expansion_count)/isochain_count[:9],
                                    f'Ending in Recovery Proportion': 2 * np.sqrt(iso_chain_recovery_count)/isochain_count[:9]})
proportionbarploterror.set_index('Length',inplace=True)
                              
fig, ax = plt.subplots(dpi=300)

proportionbarplotdf.plot(ax=ax,
                kind='bar',
                xlabel='Length of repeating pattern',
                ylabel='Proportion',
                rot=0,
                legend=False,
                yerr=proportionbarploterror,
                )
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %% 
# Finding what ends a compound chain
compchain_len_growth_1 = np.insert(compchain_len_growth,0,1)
compchain_count_growth_1 = np.insert(compchain_count_growth,0,0)
barplotdf = pd.DataFrame({'Length':compchain_len[compchain_len<10],
                            f'All Count ({len(compchains)})':compchain_count[compchain_len<10],
                            f'Ending in Growth Count ({len(compchains_growth)})':compchain_count_growth_1[compchain_len_growth_1<10],
                            f'Ending in Convection Count ({len(compchains_convec)})':compchain_count_convection[compchain_len_convection<10]})
barplotdf.set_index('Length',inplace=True)

fig, ax = plt.subplots(dpi=300)

barplotdf.plot(ax=ax,
                kind='bar',
                xlabel='Length of repeating pattern',
                ylabel='Counts',
                rot=0,
                legend=True,
                )

proportionbarplotdf = pd.DataFrame({'Length':compchain_len[compchain_len<10],
                            f'Ending in Growth Proportion':compchain_count_growth_1[compchain_len_growth_1<10]/compchain_count[:9],
                            f'Ending in Convection Proportion':compchain_count_convection[compchain_len_convection<10]/compchain_count[:9]})
proportionbarplotdf.set_index('Length',inplace=True)

proportionbarploterror = pd.DataFrame({'Length':compchain_len[compchain_len<10],
                                    f'Ending in Growth Proportion': 2 * np.sqrt(compchain_count_growth_1[compchain_len_growth_1<10])/compchain_count[:9],
                                    f'Ending in Convection Proportion': 2 * np.sqrt(compchain_count_convection[compchain_len_convection<10])/compchain_count[:9]})
proportionbarploterror.set_index('Length',inplace=True)

fig, ax = plt.subplots(dpi=300)

proportionbarplotdf.plot(ax=ax,
                kind='bar',
                xlabel='Length of repeating pattern',
                ylabel='Proportion',
                rot=0,
                legend=False,
                yerr=proportionbarploterror,
                )
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# %% 
# Plot Isolated chain length transition probability

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(isochain_len,isochain_dens,label='# of chains: {}, # of onsets: {}, max chain length: {}'.format(np.sum(isochain_count),len(iso_onsets),np.max(isochains)))
ax.errorbar(isochain_len,isochain_dens,yerr=isochain_dens_err,fmt='none',ecolor='k',capsize=2)
ax.set_title('Isolated Chains')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)


ax1.scatter(isochain_len[1:], isochain_ratio)
ax1.errorbar(isochain_len[1:], isochain_ratio, yerr=isochain_ratio_err, fmt='none', ecolor='r', capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,1.1)
ax1.fill_between(isochain_len[1:], isochain_ratio[0]-isochain_ratio_err[0], isochain_ratio[0]+isochain_ratio_err[0], alpha=0.3)
ax1.hlines(0.5,0,np.max(isochains),linestyles='dashed',color='k')
ax1.set_xlim(0,9.5)

fig.tight_layout(pad=1)

# Plot Compound chain length transition probability
fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(compchain_len,compchain_dens,label='# of chains: {}, # of onsets: {}, max chain length: {}'.format(np.sum(compchain_count),len(comp_onsets),np.max(compchains)))
ax.errorbar(compchain_len,compchain_dens,yerr=compchain_dens_err,fmt='none',ecolor='k',capsize=2)
ax.set_title('Compound Chains')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)

ax1.scatter(compchain_len[1:], compchain_ratio)
ax1.errorbar(compchain_len[1:], compchain_ratio, yerr=compchain_ratio_err, fmt='none', ecolor='r', capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,1.1)
ax1.fill_between(compchain_len[2:], compchain_ratio[1]-compchain_ratio_err[1], compchain_ratio[1]+compchain_ratio_err[1], alpha=0.3)
ax1.hlines(0.5,0,np.max(compchains),linestyles='dashed',color='k')
ax1.set_xlim(0,9.5)

fig.tight_layout(pad=1)

# %%
# Plot Compound chain length transition probability (Growth and Convection)

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(compchain_len_growth,compchain_dens_growth,label='# of chains: {}, # of onsets: {}, max chain length: {}'.format(np.sum(compchain_count_growth),len(comp_onsets_growth),np.max(compchains_growth)))
ax.errorbar(compchain_len_growth,compchain_dens_growth,yerr=compchain_dens_err_growth,fmt='none',ecolor='k',capsize=2)
ax.set_title('Compound Chains ending in Growth')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)

ax1.scatter(compchain_len_growth[1:], compchain_ratio_growth)
ax1.errorbar(compchain_len_growth[1:], compchain_ratio_growth, yerr=compchain_ratio_err_growth, fmt='none', ecolor='r',capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,1.1)
ax1.fill_between(compchain_len_growth[1:], compchain_ratio_growth[0]-compchain_ratio_err_growth[0], compchain_ratio_growth[0]+compchain_ratio_err_growth[0], alpha=0.3)
ax1.hlines(0.5,0,np.max(compchains_growth),linestyles='dashed',color='k')
ax1.set_xlim(0,9.5)

fig.tight_layout(pad=1)

fig, (ax, ax1) = plt.subplots(2,1,sharex=True,dpi=300)

ax.bar(compchain_len_convection,compchain_dens_convection,label='# of chains: {}, # of onsets: {}, max chain length: {}'.format(np.sum(compchain_count_convection),len(comp_onsets_convec),np.max(compchains_convec)))
ax.errorbar(compchain_len_convection,compchain_dens_convection,yerr=compchain_dens_err_convection,fmt='none',ecolor='k',capsize=2)
ax.set_title('Compound Chains ending in Convection')
ax.legend(loc='center right')
ax.set_xlabel('Length of repeating pattern')
ax.set_ylabel('Probability Density')
ax.xaxis.set_tick_params(labelbottom=True)

ax1.scatter(compchain_len_convection[1:], compchain_ratio_convection)
ax1.errorbar(compchain_len_convection[1:], compchain_ratio_convection, yerr=compchain_ratio_err_convection, fmt='none', ecolor='r',capsize=2, label='3 sigma error')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel("Length of repeating pattern")
ax1.set_ylabel('Transition Probability')
ax1.set_ylim(0,1.1)
ax1.fill_between(compchain_len_convection[1:], compchain_ratio_convection[0]-compchain_ratio_err_convection[0], compchain_ratio_convection[0]+compchain_ratio_err_convection[0], alpha=0.3)
ax1.hlines(0.5,0,np.max(compchains_convec),linestyles='dashed',color='k')
ax1.set_xlim(0,9.5)

fig.tight_layout(pad=1)

# %%
# Phase lengths boxplots for Isolated chains of length 1 to 9

chain_lens = np.arange(1,10)
iso_growth_dur = []
iso_exp_dur = []
iso_rec_dur = []

for index, value in enumerate(chain_lens):
    loopchainloc = np.where(isochains==value)[0]
    loop_growth_dur =[]
    loop_exp_dur = []
    loop_rec_dur = []
    for i, val in enumerate(loopchainloc):
        loopdf = sophiedf.iloc[np.where(sophiedf['Date_UTC'] == iso1stdf.iloc[val]['Date_UTC'])[0][0]-1:np.where(sophiedf['Date_UTC'] == isolastdf.iloc[val]['Date_UTC'])[0][0]+2]
        loop_growth_dur.append(loopdf[loopdf['Phase']==1]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_exp_dur.append(loopdf[loopdf['Phase']==2]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_rec_dur.append(loopdf[loopdf['Phase']==3]['Duration'].to_numpy()/np.timedelta64(1,'m'))

    iso_growth_dur.append(np.concatenate(loop_growth_dur))
    iso_exp_dur.append(np.concatenate(loop_exp_dur))
    iso_rec_dur.append(np.concatenate(loop_rec_dur))

# Plotting
fig, ax = plt.subplots(3,1, dpi=300)

ax[0].boxplot(iso_growth_dur,sym='')
ax[0].set_ylabel('Growth Duration (min)')
ax[0].set_ylim(0,960)
ax[0].set_yticks(range(0,1080,120))

ax[1].boxplot(iso_exp_dur,sym='')
ax[1].set_ylabel('Expansion Duration (min)')
ax[1].set_ylim(0,75)
ax[1].set_yticks(range(0,80,15))

ax[2].boxplot(iso_rec_dur,sym='')
ax[2].set_ylabel('Recovery Duration (min)')
ax[2].set_ylim(0,120)
ax[2].set_yticks(range(0,150,30))
ax[2].set_xlabel('Length of Isolated Chain')

fig.tight_layout(pad=1)

# %%
# Similarity of distributions of phase lengths for isolated chains of different lengths

for i, value in enumerate(iso_growth_dur[:-1]):
    print('KS test for growth durations of isolated chains of length {} and {}: {}'.format(i+2,i+1, stats.ks_2samp(iso_growth_dur[i+1],iso_growth_dur[i])))

for i, value in enumerate(iso_exp_dur[:-1]):
    print('KS test for expansions durations of isolated chains of length {} and {}: {}'.format(i+2,i+1, stats.ks_2samp(iso_exp_dur[i+1],iso_exp_dur[i])))

for i, value in enumerate(iso_rec_dur[:-1]):
    print('KS test for recovery durations of isolated chains of length {} and {}: {}'.format(i+2,i+1, stats.ks_2samp(iso_rec_dur[i+1],iso_rec_dur[i])))

# %%
# Phase lengths boxplots for compound chains of length 1 to 9

chain_lens = np.arange(2,10)
comp_growth_dur = []
comp_exp_dur = []
comp_rec_dur = []

for index, value in enumerate(chain_lens):
    loopchainloc = np.where(compchains==value)[0]
    loop_growth_dur =[]
    loop_exp_dur = []
    loop_rec_dur = []
    for i, val in enumerate(loopchainloc):
        loopdf = sophiedf.iloc[np.where(sophiedf['Date_UTC'] == comp1stdf.iloc[val]['Date_UTC'])[0][0]-1:np.where(sophiedf['Date_UTC'] == complastdf.iloc[val]['Date_UTC'])[0][0]+2]
        loop_growth_dur.append(loopdf[loopdf['Phase']==1]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_exp_dur.append(loopdf[loopdf['Phase']==2]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_rec_dur.append(loopdf[loopdf['Phase']==3]['Duration'].to_numpy()/np.timedelta64(1,'m'))

    comp_growth_dur.append(np.concatenate(loop_growth_dur))
    comp_exp_dur.append(np.concatenate(loop_exp_dur))
    comp_rec_dur.append(np.concatenate(loop_rec_dur))


# Plotting
fig, ax = plt.subplots(3,1, dpi=300)

ax[0].boxplot(comp_growth_dur,sym='')
ax[0].set_ylabel('Growth Duration (min)')
ax[0].set_ylim(0,960)
ax[0].set_yticks(range(0,1000,120))
ax[0].set_xticklabels(range(2,10))


ax[1].boxplot(comp_exp_dur,sym='')
ax[1].set_ylabel('Expansion Duration (min)')
ax[1].set_ylim(0,75)
ax[1].set_yticks(range(0,80,15))
ax[1].set_xticklabels(range(2,10))


ax[2].boxplot(comp_rec_dur,sym='')
ax[2].set_ylabel('Recovery Duration (min)')
ax[2].set_ylim(0,120)
ax[2].set_yticks(range(0,150,30))
ax[2].set_xlabel('Length of Compound Chain')
ax[2].set_xticklabels(range(2,10))

fig.tight_layout(pad=1)

# %%
# Similarity of distributions of phase lengths for compound chains of different lengths

for i, value in enumerate(comp_growth_dur[:-1]):
    print('KS test for growth durations of compound chains of length {} and {}: {}'.format(i+3,i+2, stats.ks_2samp(comp_growth_dur[i+1],comp_growth_dur[i])))

for i, value in enumerate(comp_exp_dur[:-1]):
    print('KS test for expansions durations of compound chains of length {} and {}: {}'.format(i+3,i+2, stats.ks_2samp(comp_exp_dur[i+1],comp_exp_dur[i])))

for i, value in enumerate(comp_rec_dur[:-1]):
    print('KS test for recovery durations of compound chains of length {} and {}: {}'.format(i+3,i+2, stats.ks_2samp(comp_rec_dur[i+1],comp_rec_dur[i])))

# %%
# Phase lengths boxplots for compound chains of length 2 to 9 (Growth)

chain_lens = np.arange(2,10)
comp_growth_dur = []
comp_exp_dur = []
comp_rec_dur = []

for index, value in enumerate(chain_lens):
    loopchainloc = np.where(compchains_growth==value)[0]
    loop_growth_dur =[]
    loop_exp_dur = []
    loop_rec_dur = []
    for i, val in enumerate(loopchainloc):
        loopdf = sophiedf.iloc[np.where(sophiedf['Date_UTC'] == comp1stdf.iloc[val]['Date_UTC'])[0][0]-1:np.where(sophiedf['Date_UTC'] == complastdf.iloc[val]['Date_UTC'])[0][0]+2]
        loop_growth_dur.append(loopdf[loopdf['Phase']==1]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_exp_dur.append(loopdf[loopdf['Phase']==2]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_rec_dur.append(loopdf[loopdf['Phase']==3]['Duration'].to_numpy()/np.timedelta64(1,'m'))

    comp_growth_dur.append(np.concatenate(loop_growth_dur))
    comp_exp_dur.append(np.concatenate(loop_exp_dur))
    comp_rec_dur.append(np.concatenate(loop_rec_dur))

# Plotting
fig, ax = plt.subplots(3,1, dpi=300)

ax[0].boxplot(comp_growth_dur,sym='')
ax[0].set_ylabel('Growth Duration (min)')
ax[0].set_ylim(0,960)
ax[0].set_yticks(range(0,1000,120))
ax[0].set_xticklabels(range(2,10))

ax[1].boxplot(comp_exp_dur,sym='')
ax[1].set_ylabel('Expansion Duration (min)')
ax[1].set_ylim(0,75)
ax[1].set_yticks(range(0,80,15))
ax[1].set_xticklabels(range(2,10))

ax[2].boxplot(comp_rec_dur,sym='')
ax[2].set_ylabel('Recovery Duration (min)')
ax[2].set_ylim(0,120)
ax[2].set_yticks(range(0,150,30))
ax[2].set_xlabel('Length of Compound Chain (Ended by Growth)')
ax[2].set_xticklabels(range(2,10))

fig.tight_layout(pad=1)

# %%
# Phase lengths boxplots for compound chains of length 2 to 9 (Convection)

chain_lens = np.arange(2,10)
comp_growth_dur = []
comp_exp_dur = []
comp_rec_dur = []

for index, value in enumerate(chain_lens):
    loopchainloc = np.where(compchains_convec==value)[0]
    loop_growth_dur =[]
    loop_exp_dur = []
    loop_rec_dur = []
    for i, val in enumerate(loopchainloc):
        loopdf = sophiedf.iloc[np.where(sophiedf['Date_UTC'] == comp1stdf.iloc[val]['Date_UTC'])[0][0]-1:np.where(sophiedf['Date_UTC'] == complastdf.iloc[val]['Date_UTC'])[0][0]+2]
        loop_growth_dur.append(loopdf[loopdf['Phase']==1]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_exp_dur.append(loopdf[loopdf['Phase']==2]['Duration'].to_numpy()/np.timedelta64(1,'m'))
        loop_rec_dur.append(loopdf[loopdf['Phase']==3]['Duration'].to_numpy()/np.timedelta64(1,'m'))

    comp_growth_dur.append(np.concatenate(loop_growth_dur))
    comp_exp_dur.append(np.concatenate(loop_exp_dur))
    comp_rec_dur.append(np.concatenate(loop_rec_dur))

# Plotting
fig, ax = plt.subplots(3,1, dpi=300)

ax[0].boxplot(comp_growth_dur,sym='')
ax[0].set_ylabel('Growth Duration (min)')
ax[0].set_ylim(0,960)
ax[0].set_yticks(range(0,1000,120))
ax[0].set_xticklabels(range(2,10))

ax[1].boxplot(comp_exp_dur,sym='')
ax[1].set_ylabel('Expansion Duration (min)')
ax[1].set_ylim(0,75)
ax[1].set_yticks(range(0,80,15))
ax[1].set_xticklabels(range(2,10))

ax[2].boxplot(comp_rec_dur,sym='')
ax[2].set_ylabel('Recovery Duration (min)')
ax[2].set_ylim(0,120)
ax[2].set_yticks(range(0,150,30))
ax[2].set_xlabel('Length of Compound Chain (Ended by Convection)')
ax[2].set_xticklabels(range(2,10))

fig.tight_layout(pad=1)

# %% 
# Monthly occurence of beginning of chains of length >= 2 (Isolated and Compound)

# Isolated
isochain_greq_2 = np.where(isochains>=2)[0]
isochain_greq_2_array = np.zeros(len(isochain_greq_2),dtype=int)

for i in range(len(isochain_greq_2)):
    isochain_greq_2_array[i] = iso1stdf.iloc[isochain_greq_2[i]]['Date_UTC'].month

isochain_greq_2_months, isochain_greq_2_months_count = np.unique(isochain_greq_2_array, return_counts=True)
isochain_greq_2_months_count_err = 2 * np.sqrt(isochain_greq_2_months_count)
isochain_greq_2_months_dens = isochain_greq_2_months_count/np.sum(isochain_greq_2_months_count)
isochain_greq_2_months_dens_err = isochain_greq_2_months_count_err/np.sum(isochain_greq_2_months_count)

# Compound  
compchain_greq_2 = np.where(compchains>=2)[0]
compchain_greq_2_array = np.zeros(len(compchain_greq_2),dtype=int)

for i in range(len(compchain_greq_2)):
    compchain_greq_2_array[i] = comp1stdf.iloc[compchain_greq_2[i]]['Date_UTC'].month

compchain_greq_2_months, compchain_greq_2_months_count = np.unique(compchain_greq_2_array, return_counts=True)
compchain_greq_2_months_count_err = 2 * np.sqrt(compchain_greq_2_months_count)
compchain_greq_2_months_dens = compchain_greq_2_months_count/np.sum(compchain_greq_2_months_count)
compchains_greq_2_months_dens_err = compchain_greq_2_months_count_err/np.sum(compchain_greq_2_months_count)
# Plotting
fig, ax = plt.subplots(2,1,dpi=300,sharex=True,sharey=True)

ax[0].bar(isochain_greq_2_months,isochain_greq_2_months_dens,label='Number of chains: {}'.format(len(isochain_greq_2)))
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Monthly occurence of First Onset of Isolated chains (Length >= 2)')
ax[0].legend(loc='best')
ax[0].xaxis.set_tick_params(labelbottom=True)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

ax[1].bar(compchain_greq_2_months,compchain_greq_2_months_dens,color=colormap[1], label='Number of chains: {}'.format(len(compchain_greq_2)))
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('Monthly occurence of First Onset of Compound chains (Length >= 2)')
ax[1].legend(loc='best')

fig.tight_layout(pad=1)

# Plotting with error bars

fig, ax = plt.subplots(dpi=300)

ax.errorbar(isochain_greq_2_months,isochain_greq_2_months_dens,yerr=isochain_greq_2_months_dens_err,fmt=colormap[0],ecolor=colormap[0],capsize=2,label='Isolated, # of chains: {}'.format(len(isochain_greq_2)))
ax.errorbar(compchain_greq_2_months,compchain_greq_2_months_dens,yerr=compchains_greq_2_months_dens_err,fmt=colormap[1], ecolor=colormap[1],capsize=2,label='Compound # of chains: {}'.format(len(compchain_greq_2)))
ax.set_xlabel('Month')
ax.set_ylabel('Probability Density')
ax.set_title('Monthly occurence of First Onset of chains (Length >= 2)')
ax.legend(loc='best')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_ylim(0,0.15)

fig.tight_layout(pad=1)

# %%
# Monthly occurence of beginning of chains of length >= 2 (Compound ending in Growth and Convection)

compchain_greq_2_growth = np.where(compchains_growth>=2)[0]
compchain_greq_2_growth_array = np.zeros(len(compchain_greq_2_growth),dtype=int)

for i in range(len(compchain_greq_2_growth)):
    compchain_greq_2_growth_array[i] = comp1stdf.iloc[compchain_greq_2_growth[i]]['Date_UTC'].month

compchain_greq_2_growth_months, compchain_greq_2_growth_months_count = np.unique(compchain_greq_2_growth_array, return_counts=True)
compchain_greq_2_growth_months_count_err = 2 * np.sqrt(compchain_greq_2_growth_months_count)
compchain_greq_2_growth_months_dens = compchain_greq_2_growth_months_count/np.sum(compchain_greq_2_growth_months_count)
compchain_greq_2_growth_months_dens_err = compchain_greq_2_growth_months_count_err/np.sum(compchain_greq_2_growth_months_count)

compchain_greq_2_convec = np.where(compchains_convec>=2)[0]
compchain_greq_2_convec_array = np.zeros(len(compchain_greq_2_convec),dtype=int)

for i in range(len(compchain_greq_2_convec)):
    compchain_greq_2_convec_array[i] = comp1stdf.iloc[compchain_greq_2_convec[i]]['Date_UTC'].month

compchain_greq_2_convec_months, compchain_greq_2_convec_months_count = np.unique(compchain_greq_2_convec_array, return_counts=True)
compchain_greq_2_convec_months_count_err = 2 * np.sqrt(compchain_greq_2_convec_months_count)
compchain_greq_2_convec_months_dens = compchain_greq_2_convec_months_count/np.sum(compchain_greq_2_convec_months_count)
compchain_greq_2_convec_months_dens_err = compchain_greq_2_convec_months_count_err/np.sum(compchain_greq_2_convec_months_count)

fig, ax = plt.subplots(2,1,dpi=300,sharex=True,sharey=True)

ax[0].bar(compchain_greq_2_growth_months,compchain_greq_2_growth_months_dens,color=colormap[2],label='Number of chains: {}'.format(len(compchain_greq_2_growth)))
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Monthly occurence of First Onset of Compound chains ending in Growth (Length >= 2)')
ax[0].legend(loc='best')
ax[0].xaxis.set_tick_params(labelbottom=True)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

ax[1].bar(compchain_greq_2_convec_months,compchain_greq_2_convec_months_dens,color=colormap[3], label='Number of chains: {}'.format(len(compchain_greq_2_convec)))
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('Monthly occurence of First Onset of Compound chains ending in Convection (Length >= 2)')
ax[1].legend(loc='best')

fig.tight_layout(pad=1)

fig, ax = plt.subplots(dpi=300)

ax.errorbar(compchain_greq_2_growth_months,compchain_greq_2_growth_months_dens,yerr=compchain_greq_2_growth_months_dens_err,fmt=colormap[2],ecolor=colormap[2],capsize=2,label='Compound ending in Growth, # of chains: {}'.format(len(compchain_greq_2_growth)))
ax.errorbar(compchain_greq_2_convec_months,compchain_greq_2_convec_months_dens,yerr=compchain_greq_2_convec_months_dens_err,fmt=colormap[3], ecolor=colormap[3],capsize=2,label='Compound ending in Convection, # of chains: {}'.format(len(compchain_greq_2_convec)))
ax.set_xlabel('Month')
ax.set_ylabel('Probability Density')
ax.set_title('Monthly occurence of First Onset of Compound chains (Length >= 2)')
ax.legend(loc='best')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_ylim(0,0.15)

fig.tight_layout(pad=1)
# %%