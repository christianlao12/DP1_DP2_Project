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

# %% 
# Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT90_1996-2021.txt", low_memory=False)
sophiedf['Date_UTC'] = pd.to_datetime(sophiedf['Date_UTC'])
sophiedf['Duration'] = np.append(np.diff(sophiedf["Date_UTC"].to_numpy()),0)
sophiedf = sophiedf[sophiedf['Date_UTC'].between('1997','2020')].reset_index(drop=True)
sophiedf.rename(columns={'DeltaSML':'Delbay'},inplace=True)
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
first_iso = np.where(np.diff(array_iso) == 1)[0] + 1
last_iso = np.where(np.diff(array_iso) == -1)[0]
first_iso = np.append(0, first_iso)
# last_iso = np.append(last_iso,len(expansiondf)-1)
iso1stdf = expansiondf.iloc[first_iso]
isolastdf = expansiondf.iloc[last_iso]
isochains = (isolastdf.index.to_numpy() - iso1stdf.index.to_numpy()) + 1

# Compound Chains
compound = np.intersect1d(np.where(expansiondf['Compound Onset']==1),np.where(expansiondf['NewFlag']==0))
comp_onsets = expansiondf.iloc[compound]

array_comp = np.zeros(len(expansiondf))
array_comp[compound] = True
first_comp = np.where(np.diff(array_comp) == 1)[0] + 1
last_comp = np.where(np.diff(array_comp) == -1)[0]
# first_comp = first_comp[:-1]
# last_comp = np.append(last_comp,len(expansiondf)-1)
comp1stdf = expansiondf.iloc[first_comp]
subset_comp = expansiondf.iloc[np.setdiff1d(compound,first_comp)]
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


#%% 
# Loading in additional data
# Loading in OMNI Data
omnidf = pd.read_csv('Data/OMNIData.csv')
omnidf['Date_UTC'] = pd.to_datetime(omnidf['Date_UTC'])
omnidf = omnidf[omnidf['Date_UTC'].between('1998','2022')].reset_index(drop=True)
omnidf['Clock_angle'] = np.arctan2(omnidf['BY_GSM'].to_numpy(),omnidf['BZ_GSM'])*180/np.pi

#  Loading in SME Data
smedf = pd.read_csv('Data/SMEdata.txt')
smedf['Date_UTC'] = pd.to_datetime(smedf['Date_UTC'])
smedf['SML'].replace(999999, np.nan, inplace=True)
smedf['SMU'].replace(999999, np.nan, inplace=True)
smedf = smedf[smedf['Date_UTC'].between('1998','2022')].reset_index(drop=True)

#  Loading in SMR Data
smrdf = pd.read_csv('Data/SMRdata.txt')
smrdf['Date_UTC'] = pd.to_datetime(smrdf['Date_UTC'])
smrdf['SMR'].replace(999999, np.nan, inplace=True)
smrdf['SMR00'].replace(999999, np.nan, inplace=True)
smrdf['SMR06'].replace(999999, np.nan, inplace=True)
smrdf['SMR12'].replace(999999, np.nan, inplace=True)
smrdf['SMR18'].replace(999999, np.nan, inplace=True)
smrdf = smrdf[smrdf['Date_UTC'].between('1998','2022')].reset_index(drop=True)


# %% 
# Isolated chain length vs IMF Bz (30 mins before onset)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq) 
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Bz in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs IMF Bz (30 mins before onset) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Bz in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs IMF Bz (30 mins before onset) (Growth)

x_edges = np.arange(2,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []   
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_growth==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length (Ends in Growth)")
ax.set_ylabel("Bz in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# Compound chain length vs IMF Bz (30 mins before onset) (Convection)
chain_vals = []
chain_hist = []
chain_median = []   
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_convec==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length (Ends in Convection)")
ax.set_ylabel("Bz in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Isolated chain length vs vsw (30 mins before onset)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Vsw in 30 mins prior to first onset (km/s)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs vsw (30 mins before onset) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Vsw in 30 mins prior to first onset (km/s)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs vsw (30 mins before onset) (Growth)

x_edges = np.arange(2,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_growth==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length (Ends in Growth)")
ax.set_ylabel("Vsw in 30 mins prior to first onset (km/s)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# Compound chain length vs vsw (30 mins before onset) (Convection)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_convec==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length (Ends in Convection)")
ax.set_ylabel("Vsw in 30 mins prior to first onset (km/s)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Iso chain length vs SMR (30 mins before onset)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-40,12.5,2.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("SMR in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs SMR (30 mins before onset) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-40,12.5,2.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("SMR in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

fig.tight_layout(pad=1)

# %%
# Isolated chain length vs SMU (30 mins before onset)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(0,210,10)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = iso1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(smedf[smedf['Date_UTC'].between(start,end,inclusive='left')]['SMU'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("SMU in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain length vs SMU (30 mins before onset) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(0,210,10)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for j in range(len(chain_len_index)):
        end = comp1stdf.iloc[chain_len_index[j]]['Date_UTC']
        start = end - pd.to_timedelta(.5,'h')
    
        chain_vals_loop.append(smedf[smedf['Date_UTC'].between(start,end,inclusive='left')]['SMU'].to_numpy())
    
    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("SMU in 30 mins prior to first onset (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Isolated chain vs IMF Bz (30 mins prior to end of last recovery phase)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Bz in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Compound chain vs IMF Bz (30 mins prior to end of last recovery phase) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Bz in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Compound chain vs IMF Bz (30 mins prior to end of last recovery phase) (Growth)

x_edges = np.arange(2,11) - .5
y_edges = np.arange(-5,5.5,.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_growth==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length (Ends in Growth)")
ax.set_ylabel("Bz in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# Compound chain vs IMF Bz (30 mins prior to end of last recovery phase) (Convection)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_convec==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['BZ_GSM'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.25)
ax.set_xlabel("Compound Chain Length (Ends in Convection)")
ax.set_ylabel("Bz in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

fig.tight_layout(pad=1)

# %%
# Isolated chain vs vsw (30 mins before end of last recovery phase)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("Vsw in 30 mins prior to end of last recovery phase (km/s)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain vs vsw (30 mins before end of last recovery phase) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("Vsw in 30 mins prior to end of last recovery phase (km/s)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain vs vsw (30 mins before end of last recovery phase) (Growth)

x_edges = np.arange(2,11) - .5
y_edges = np.arange(250,825,25)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_growth==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length (Ends in Growth)")
ax.set_ylabel("Vsw in 30 mins prior to end of last recovery phase (km/s)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# Compound chain vs vsw (30 mins before end of last recovery phase) (Convection)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(2,10):
    chain_len_index = np.where(compchains_convec==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(omnidf[omnidf['Date_UTC'].between(start,end,inclusive='left')]['flow_speed'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.3)
ax.set_xlabel("Compound Chain Length (Ends in Convection)")
ax.set_ylabel("Vsw in 30 mins prior to end of last recovery phase (km/s)")
ax.scatter(np.arange(2,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(2,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Isolated chain vs SMR (30 mins before end of last recovery phase)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-40,12.5,2.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("SMR in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

fig.tight_layout(pad=1)

# %%
#  Compound chain vs SMR (30 mins before end of last recovery phase) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(-40,12.5,2.5)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(smrdf[smrdf['Date_UTC'].between(start,end,inclusive='left')]['SMR'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("SMR in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

fig.tight_layout(pad=1)

# %%
# Isolated chain vs SMU (30 mins before end of last recovery phase)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(0,260,10)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(isochains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==isolastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(smedf[smedf['Date_UTC'].between(start,end,inclusive='left')]['SMU'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Isolated Chain Length")
ax.set_ylabel("SMU in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
# Compound chain vs SMU (30 mins before end of last recovery phase) (All Compound)

x_edges = np.arange(1,11) - .5
y_edges = np.arange(0,260,10)

chain_vals = []
chain_hist = []
chain_median = []
chain_lq = []
chain_uq = []

for i in range(1,10):
    chain_len_index = np.where(compchains==i)[0]
    chain_vals_loop = []

    for __, j in enumerate(chain_len_index):
        end = sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Date_UTC'] + sophiedf.iloc[np.where(sophiedf['Date_UTC']==complastdf.iloc[j]['Date_UTC'])[0][0]+1]['Duration']
        start = end - pd.to_timedelta(.5,'h')
        
        chain_vals_loop.append(smedf[smedf['Date_UTC'].between(start,end,inclusive='left')]['SMU'].to_numpy())

    chain_vals_loop = np.concatenate(chain_vals_loop)
    chain_hist_loop = np.histogram(chain_vals_loop,bins=y_edges)[0]
    chain_hist_loop = chain_hist_loop/np.sum(chain_hist_loop)
    chain_median_loop = np.nanmean(chain_vals_loop)
    chain_lq_loop = np.nanquantile(chain_vals_loop,0.25)
    chain_uq_loop = np.nanquantile(chain_vals_loop,0.75)

    chain_vals.append(chain_vals_loop)
    chain_hist.append(chain_hist_loop)
    chain_median.append(chain_median_loop)
    chain_lq.append(chain_lq_loop)
    chain_uq.append(chain_uq_loop)

chain_lq_diff =  np.array(chain_median) - np.array(chain_lq)
chain_uq_diff = np.array(chain_uq) - np.array(chain_median)
chain_quantiles = np.array([chain_lq_diff,chain_uq_diff])

X,Y = np.meshgrid(x_edges,y_edges)
values = np.array(chain_hist)

fig, ax = plt.subplots(dpi=300)

plot = ax.pcolormesh(X,Y,values.T,cmap='viridis',vmax=0.2)
ax.set_xlabel("Compound Chain Length")
ax.set_ylabel("SMU in 30 mins prior to end of last recovery phase (nT)")
ax.scatter(np.arange(1,10),chain_median,c=colormap[3],marker='x',label='Median')
ax.errorbar(np.arange(1,10),chain_median,yerr=chain_quantiles,fmt='none',ecolor=colormap[3],elinewidth=1,capsize=2,label='Quantile Range')
plt.colorbar(plot, ax=ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

fig.tight_layout(pad=1)

# %%
