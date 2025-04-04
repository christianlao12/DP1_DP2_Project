# %% # Importing Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

# Housekeeping
sns.set_theme(context="notebook",style="whitegrid",palette="colorblind")
colormap = sns.color_palette("colorblind")

# Labelling of the MLT bins
bins = np.roll(np.arange(0, 24), 12)
bins = list(map(str, bins))

# Function to calculate chi squared test
def chi_squared_test(measured, model, uncertainty):
    return np.sum(np.nan_to_num((np.array(measured) - np.array(model)) ** 2 / np.array(uncertainty) ** 2))

# %% Loading in Substorm Data

# Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT90_1996-2024_NewFlag.csv")
sophiedf["Date_UTC"] = pd.to_datetime(sophiedf["Date_UTC"])
sophiedf["Duration"] = np.append(np.diff(sophiedf["Date_UTC"].to_numpy()), 0)
sophiedf = sophiedf[sophiedf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)
# sophiedf = sophiedf[sophiedf["Date_UTC"].between("2000-05-19", "2003-01-01", inclusive="left")].reset_index(drop=True)
if "Delbay" in sophiedf.columns:
    sophiedf.rename(columns={"Delbay": "DeltaSML"}, inplace=True)
if "SML Val at End" in sophiedf.columns:
    sophiedf.rename(columns={"SML Val at End": "SMLatEnd"}, inplace=True)

sophiedf["DeltaSML"] = pd.to_numeric(sophiedf["DeltaSML"], errors="coerce")
sophiedf = sophiedf.loc[2:].reset_index(drop=True)

sophiedf['Flag'] = sophiedf['Flag'].apply(lambda x: 1 if x > 0 else 0)

# Loading in Frey Data
freydf = pd.read_csv("Data/FreySubstorms.csv", low_memory=False)
freydf["Date_UTC"] = pd.to_datetime(freydf["Date_UTC"])
fredydf = freydf[freydf["Date_UTC"].between("2000-05-19", "2003-01-01", inclusive="left")].reset_index(drop=True)

# Loading in Newell and Ohtani onsets
newelldf = pd.read_csv("Data/NewellSubstorms_1997_2022.csv")
newelldf["Date_UTC"] = pd.to_datetime(newelldf["Date_UTC"])
newelldf = newelldf[newelldf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)
# newelldf = newelldf[newelldf["Date_UTC"].between("2000-05-19", "2003-01-01", inclusive="left")].reset_index(drop=True)

ohtanidf = pd.read_csv("Data/OhtaniSubstorms_1997_2022.csv")
ohtanidf["Date_UTC"] = pd.to_datetime(ohtanidf["Date_UTC"])
ohtanidf = ohtanidf[ohtanidf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)
# ohtanidf = ohtanidf[ohtanidf["Date_UTC"].between("2000-05-19", "2003-01-01", inclusive="left")].reset_index(drop=True)

# Load Shore MLT Data
shoredp1 = pd.read_csv("Data/ShoreData/ShoreSpatialAmpDP1.csv")
shoredp2 = pd.read_csv("Data/ShoreData/ShoreSpatialAmpDP2.csv")

# %% SOPHIE Event Classification
# Isolated Onsets
iso_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
for i in range(1, len(sophiedf["Date_UTC"]) - 2):
    if (
        (sophiedf.iloc[i - 1]["Phase"] == 1)
        and (sophiedf.iloc[i]["Phase"] == 2)
        and (sophiedf.iloc[i + 1]["Phase"] == 3)
        and (sophiedf.iloc[i + 2]["Phase"] == 1)
    ):
        iso_arr[i] = 1  # GERG
sophiedf["Isolated"] = iso_arr

# Compound Onsets
comp_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
for i in range(2, len(sophiedf["Date_UTC"]) - 1):
    if (
        (sophiedf.iloc[i - 2]["Phase"] == 2)
        and (sophiedf.iloc[i - 1]["Phase"] == 3)
        and (sophiedf.iloc[i]["Phase"] == 2)
    ):
        comp_arr[i] = 1  # Compound Onset

for i, val in enumerate(comp_arr):
    if val == 1:
        comp_arr[i - 2] = 1
sophiedf["Compound"] = comp_arr

# Flagging Onsets after Convection intervals
newflag_arr = sophiedf["Flag"].to_numpy().copy()
for i in range(1, len(sophiedf["Flag"])):
    if newflag_arr[i] == 1 or (
        newflag_arr[i - 1] == 1 and sophiedf.iloc[i]["Phase"] != 1
    ):
        newflag_arr[i] = 1
sophiedf["NewFlag"] = newflag_arr

# %% SOPHIE Event types

# Only Expansion Phases
expansiondf = sophiedf.iloc[np.where(sophiedf["Phase"] == 2)].reset_index(drop=True)

# Only Convection Expansion
flag_id = np.intersect1d(np.where(expansiondf['Phase'] == 2), np.where(expansiondf["Flag"] == 1))
convec_expansiondf = expansiondf.iloc[flag_id]

# Substorm Expansions
nonflag_id = np.intersect1d(np.where(expansiondf['Phase'] == 2), np.where(expansiondf["Flag"] == 0))
substorm_expansion = sophiedf.iloc[nonflag_id]

# Isolated Onsets
iso_id = np.intersect1d(np.where(expansiondf["Isolated"] == 1), np.where(expansiondf["Flag"] == 0))
iso_onsets = expansiondf.iloc[iso_id]

# Compound Onsets
comp_id = np.intersect1d(np.where(expansiondf["Compound"] == 1), np.where(expansiondf["NewFlag"] == 0))
comp_onsets = expansiondf.iloc[comp_id]

# After Convection Onsets
after_convec_id = np.intersect1d(np.where(expansiondf["Phase"] == 2), np.setdiff1d(np.where(expansiondf["NewFlag"] == 1), np.where(expansiondf["Flag"] == 1)))
after_convec_onsets = expansiondf.iloc[after_convec_id]

# Other Onsets
other_id = np.setdiff1d(nonflag_id, np.concatenate([iso_id, comp_id, after_convec_id]))
other_onsets = expansiondf.iloc[other_id]

# %% Analysing seasonal variations of Other Onsets

otheryear = other_onsets["Date_UTC"].dt.year
othermonths = other_onsets["Date_UTC"].dt.month
otherhours = other_onsets["Date_UTC"].dt.hour

# Plotting yearly variations
fig, ax = plt.subplots(dpi=300)
ax.hist(otheryear,bins=np.arange(1997,2021), density=False,label="Other substorms ", histtype="step",color=colormap[0])
ax.set_xlabel("Year")
ax.set_ylabel("Counts")
ax.set_xticks(range(1997,2021,2))
ax.legend(loc='best')
fig.tight_layout(pad=1)

# Plotting monthly variations
fig, ax = plt.subplots(dpi=300)
ax.hist(othermonths,bins=np.arange(1,14), density=False,label="Other substorms ", histtype="step",color=colormap[0])
ax.set_xlabel("Month")
ax.set_ylabel("Counts")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(loc='best')
fig.tight_layout(pad=1)

# Plotting hourly variations
fig, ax = plt.subplots(dpi=300)
ax.hist(otherhours,bins=np.arange(0,25), density=False,label="Other substorms ", histtype="step",color=colormap[0])
ax.set_xlabel("Hour")
ax.set_ylabel("Counts")
ax.set_xticks(range(0,25,2))
ax.legend(loc='best')
fig.tight_layout(pad=1)


# %% Substorm MLT distributions

# All onsets
onsets_mlt = sophiedf.iloc[np.where(sophiedf["Phase"] == 2)]["MLT"].to_numpy()

onsets_mlt_counts, onsets_mlt_bins = np.histogram(onsets_mlt, bins=np.arange(0, 25))
onsets_mlt_bins = onsets_mlt_bins[:-1]
onsets_mlt_counts = [*onsets_mlt_counts[12:], *onsets_mlt_counts[:12]]
onsets_mlt_counts_err = 2 * np.sqrt(onsets_mlt_counts)
onsets_mlt_dens = onsets_mlt_counts / np.sum(onsets_mlt_counts)
onsets_mlt_dens_err = onsets_mlt_counts_err / np.sum(onsets_mlt_counts)

# Substorms
substorm_onsets_mlt = expansiondf.iloc[np.where(expansiondf["Flag"] == 0)]["MLT"].to_numpy()

substorm_onsets_mlt_counts, substorm_onsets_mlt_bins = np.histogram(substorm_onsets_mlt, bins=np.arange(0, 25))
substorm_onsets_mlt_bins = substorm_onsets_mlt_bins[:-1]
substorm_onsets_mlt_counts = [*substorm_onsets_mlt_counts[12:], *substorm_onsets_mlt_counts[:12]]
substorm_onsets_mlt_counts_err = 2 * np.sqrt(substorm_onsets_mlt_counts)
substorm_onsets_mlt_dens = substorm_onsets_mlt_counts / np.sum(substorm_onsets_mlt_counts)
substorm_onsets_mlt_dens_err = substorm_onsets_mlt_counts_err / np.sum(substorm_onsets_mlt_counts)

# Isolated
isolated_onsets_mlt = iso_onsets["MLT"].to_numpy()

iso_mlt_counts, iso_mlt_bins = np.histogram(isolated_onsets_mlt, bins=np.arange(0, 25))
iso_mlt_bins = iso_mlt_bins[:-1]
iso_mlt_counts = [*iso_mlt_counts[12:], *iso_mlt_counts[:12]]
iso_mlt_counts_err = 2 * np.sqrt(iso_mlt_counts)
iso_mlt_dens = iso_mlt_counts / np.sum(iso_mlt_counts)
iso_mlt_dens_err = iso_mlt_counts_err / np.sum(iso_mlt_counts)

# Compound
compound_onsets_mlt = comp_onsets["MLT"].to_numpy()
comp_mlt_counts, comp_mlt_bins = np.histogram(compound_onsets_mlt, bins=np.arange(0, 25))
comp_mlt_bins = comp_mlt_bins[:-1]
comp_mlt_counts = [*comp_mlt_counts[12:], *comp_mlt_counts[:12]]
comp_mlt_counts_err = 2 * np.sqrt(comp_mlt_counts)
comp_mlt_dens = comp_mlt_counts / np.sum(comp_mlt_counts)
comp_mlt_dens_err = comp_mlt_counts_err / np.sum(comp_mlt_counts)

# Convection expansions
convec_onsets_mlt = convec_expansiondf["MLT"].to_numpy()
convec_mlt_counts, convec_mlt_bins = np.histogram(convec_onsets_mlt, bins=np.arange(0, 25))
convec_mlt_bins = convec_mlt_bins[:-1]
convec_mlt_counts = [*convec_mlt_counts[12:], *convec_mlt_counts[:12]]
convec_mlt_counts_err = 2 * np.sqrt(convec_mlt_counts)
convec_mlt_dens = convec_mlt_counts / np.sum(convec_mlt_counts)
convec_mlt_dens_err = convec_mlt_counts_err / np.sum(convec_mlt_counts)

# After convection expansions
after_convec_mlt = after_convec_onsets["MLT"].to_numpy()
after_convec_mlt_counts, after_convec_mlt_bins = np.histogram(after_convec_mlt, bins=np.arange(0, 25))
after_convec_mlt_bins = after_convec_mlt_bins[:-1]
after_convec_mlt_counts = [*after_convec_mlt_counts[12:], *after_convec_mlt_counts[:12]]
after_convec_mlt_counts_err = 2 * np.sqrt(after_convec_mlt_counts)
after_convec_mlt_dens = after_convec_mlt_counts / np.sum(after_convec_mlt_counts)
after_convec_mlt_dens_err = after_convec_mlt_counts_err / np.sum(after_convec_mlt_counts)

# Other expansions
other_mlt = other_onsets["MLT"].to_numpy()
other_mlt_counts, other_mlt_bins = np.histogram(other_mlt, bins=np.arange(0, 25))
other_mlt_bins = other_mlt_bins[:-1]
other_mlt_counts = [*other_mlt_counts[12:], *other_mlt_counts[:12]]
other_mlt_counts_err = 2 * np.sqrt(other_mlt_counts)
other_mlt_dens = other_mlt_counts / np.sum(other_mlt_counts)
other_mlt_dens_err = other_mlt_counts_err / np.sum(other_mlt_counts)

# Total of Isolated, Compound and Convection
total_mlt_counts = (np.array(iso_mlt_counts) + np.array(comp_mlt_counts) + np.array(convec_mlt_counts))
total_mlt_counts_err = 2 * np.sqrt(total_mlt_counts)
total_mlt_dens = total_mlt_counts / np.sum(total_mlt_counts)
total_mlt_dens_err = total_mlt_counts_err / np.sum(total_mlt_counts)

# Frey MLT
frey_mlt = freydf["MLT"].to_numpy()
frey_mlt_counts, frey_mlt_bins = np.histogram(frey_mlt, bins=np.arange(0, 25))
frey_mlt_bins = frey_mlt_bins[:-1]
frey_mlt_counts = [*frey_mlt_counts[12:], *frey_mlt_counts[:12]]
frey_mlt_counts_err = 2 * np.sqrt(frey_mlt_counts)
frey_mlt_dens = frey_mlt_counts / np.sum(frey_mlt_counts)
frey_mlt_dens_err = frey_mlt_counts_err / np.sum(frey_mlt_counts)

# Newell MLT
newell_mlt = newelldf["MLT"].to_numpy()
newell_mlt_counts, newell_mlt_bins = np.histogram(newell_mlt, bins=np.arange(0, 25))
newell_mlt_bins = newell_mlt_bins[:-1]
newell_mlt_counts = [*newell_mlt_counts[12:], *newell_mlt_counts[:12]]
newell_mlt_counts_err = 2 * np.sqrt(newell_mlt_counts)
newell_mlt_dens = newell_mlt_counts / np.sum(newell_mlt_counts)
newell_mlt_dens_err = newell_mlt_counts_err / np.sum(newell_mlt_counts)

# Ohtani MLT
ohtani_mlt = ohtanidf["MLT"].to_numpy()
ohtani_mlt_counts, ohtani_mlt_bins = np.histogram(ohtani_mlt, bins=np.arange(0, 25))
ohtani_mlt_bins = ohtani_mlt_bins[:-1]
ohtani_mlt_counts = [*ohtani_mlt_counts[12:], *ohtani_mlt_counts[:12]]
ohtani_mlt_counts_err = 2 * np.sqrt(ohtani_mlt_counts)
ohtani_mlt_dens = ohtani_mlt_counts / np.sum(ohtani_mlt_counts)
ohtani_mlt_dens_err = ohtani_mlt_counts_err / np.sum(ohtani_mlt_counts)

# %% Plotting Different Onset List MLT Distributions
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,onsets_mlt_counts,label="All SOPHIE Events: No. of onsets: {}".format(len(onsets_mlt)),alpha = 0.75)
ax.plot(np.arange(24) + 0.5,iso_mlt_counts,label="SOPHIE Isolated Substorms: No. of onsets: {}".format(len(substorm_onsets_mlt)),alpha = 0.75)
ax.plot(np.arange(24) + 0.5,newell_mlt_counts,label="Newell & Gjerloev 2011: No. of onsets: {}".format(len(newelldf)),ls="-.",)
ax.plot(np.arange(24) + 0.5,ohtani_mlt_counts,label="Ohtani & Gjerloev 2020: No. of onsets: {}".format(len(ohtanidf)),ls="-.",)
# ax.plot(np.arange(24) + 0.5,frey_mlt_counts,label="Frey et al. 2004: No. of onsets: {}".format(len(freydf)),ls="-.",)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

# %% Fitting MLT distributions

# Fitting Isolated from Convection distribution
chi_hist = np.inf
iso_convec_fit = np.zeros(24)
wght_hist = 0

for i in range(len(iso_onsets)):
    dist = i * np.array(convec_mlt_dens)
    dist = np.round(dist, 0)
    chi_sq = chi_squared_test(iso_mlt_counts, dist, iso_mlt_counts_err)
    
    if (np.array(iso_mlt_counts) - dist).min() >= 0:
        chi_hist = chi_sq
        wght_hist = i
        iso_convec_fit = dist

iso_minus_convec = np.array(iso_mlt_counts) - iso_convec_fit

dp1 = iso_minus_convec
dp2 = convec_mlt_counts

dp1_dens = dp1 / np.sum(dp1)
dp2_dens = dp2 / np.sum(dp2)

# Fitting to Isolated distribution by weighted sum of isolated and convection expansion

chi_iso = np.inf
fit_iso = []
n_substorm_iso = 0

for i in range(len(iso_onsets)):
    dist = np.array(dp1_dens) * i + np.array(dp2_dens) * (len(iso_onsets) - i)
    chi_sq = chi_squared_test(iso_mlt_counts, dist, iso_mlt_counts_err)
    if chi_sq < chi_iso:
        chi_iso = chi_sq
        n_substorm_iso = i
        fit_iso = dist

fit_iso_dens = fit_iso / np.sum(fit_iso)
n_convec_iso = len(iso_onsets) - n_substorm_iso

# Fitting to Compound distribution by weighted sum of isolated and convection expansion

chi_comp = np.inf
fit_comp = []
n_substorm_comp = 0

for i in range(len(comp_onsets)):
    dist = np.array(dp1_dens) * i + np.array(dp2_dens) * (len(comp_onsets) - i)
    chi_sq = chi_squared_test(comp_mlt_counts, dist, comp_mlt_counts_err)
    if chi_sq < chi_comp:
        chi_comp = chi_sq
        n_substorm_comp = i
        fit_comp = dist

fit_comp_dens = fit_comp / np.sum(fit_comp)
n_convec_comp = len(comp_onsets) - n_substorm_comp

# Fitting to After Convection distribution by weighted sum of isolated and convection expansion

chi_after_convec = np.inf
fit_after_convec = []
n_substorm_after_convec = 0

for i in range(len(after_convec_onsets)):
    dist = np.array(dp1_dens) * i + np.array(dp2_dens) * (len(after_convec_onsets) - i)
    chi_sq = chi_squared_test(after_convec_mlt_counts, dist, after_convec_mlt_counts_err)
    if chi_sq < chi_after_convec:
        chi_after_convec = chi_sq
        n_substorm_after_convec = i
        fit_after_convec = dist

fit_after_convec_dens = fit_after_convec / np.sum(fit_after_convec)
n_convec_after_convec = len(after_convec_onsets) - n_substorm_after_convec

# Fitting to Other distribution by weighted sum of isolated and convection expansion

chi_other = np.inf
fit_other = []
n_substorm_other = 0

for i in range(len(other_onsets)):
    dist = np.array(dp1_dens) * i + np.array(dp2_dens) * (len(other_onsets) - i)
    chi_sq = chi_squared_test(other_mlt_counts, dist, other_mlt_counts_err)
    if chi_sq < chi_other:
        chi_other = chi_sq
        n_substorm_other = i
        fit_other = dist

fit_other_dens = fit_other / np.sum(fit_other)
n_convec_other = len(other_onsets) - n_substorm_other

# %% Plotting Known distributions

# Plotting Histograms (Total, Isolated, Compound, Convection, Frey)
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,total_mlt_counts,color=colormap[0],label="Total: No. of events: {}".format(np.sum(total_mlt_counts)),)
ax.plot(np.arange(24) + 0.5,iso_mlt_counts,color=colormap[1],label="Isolated substorm: No. of events: {}".format(len(iso_onsets)),)
ax.plot(np.arange(24) + 0.5,comp_mlt_counts,color=colormap[3],label="Compound substorm: No. of events: {}".format(len(comp_onsets)),)
ax.plot(np.arange(24) + 0.5,convec_mlt_counts,color=colormap[2],label="Convection enhancement: No. of events: {}".format(len(convec_expansiondf)),)
ax.plot(np.arange(24) + 0.5,frey_mlt_counts,color=colormap[6],label="Frey et al. 2004 auroral substorms: No. of events: {}".format(len(freydf)),)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Plotting Probability Distributions (Total, Isolated, Compound, Convection, Frey)
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,total_mlt_dens,color=colormap[0],label="Total: No. of events: {}".format(np.sum(total_mlt_counts)),)
ax.plot(np.arange(24) + 0.5,iso_mlt_dens,color=colormap[1],label="Isolated substorm: No. of events: {}".format(len(iso_onsets)),)
ax.plot(np.arange(24) + 0.5,comp_mlt_dens,color=colormap[3],label="Compound substorm: No. of events: {}".format(len(comp_onsets)),)
ax.plot(np.arange(24) + 0.5,convec_mlt_dens,color=colormap[2],label="Convection enhancement: No. of events: {}".format(len(convec_expansiondf)),)
ax.plot(np.arange(24) + 0.5,frey_mlt_dens,color=colormap[6],label="Frey et al. 2004 auroral substorms: No. of events: {}".format(len(freydf)),)
ax.set_xlabel("MLT")
ax.set_ylabel("Probability")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Plotting Probability Distributions (Isolated, Convection)

fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,iso_mlt_dens,color=colormap[1],label="Isolated substorm: No. of events: {}".format(len(iso_onsets)),)
ax.plot(np.arange(24) + 0.5,convec_mlt_dens,color=colormap[2],label="Convection enhancement: No. of events: {}".format(len(convec_expansiondf)),)
ax.set_xlabel("MLT")
ax.set_ylabel("Probability")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Plotting Histogram (Isolated, Convection Fit, Residual)
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5, iso_mlt_counts, color=colormap[1], label="Isolated substorm")
# ax.plot(convec_mlt_counts,color=colormap[3],label='Convection Expansion Phase: No. of onsets: {}'.format(len(convec_expansiondf)))
ax.plot(np.arange(24) + 0.5,iso_convec_fit,color=colormap[2],ls="-.",label="Scaled Convection enhancement fit",)
ax.plot(np.arange(24) + 0.5,iso_minus_convec,color=colormap[7],ls="-.",label="DP1 Perturbation = Isolated substorm - Scaled Convection enhancement fit",)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Plotting Probability Distributions (Residual, Convection, Frey)
fig, ax = plt.subplots(2,1,dpi=300,sharex=True,gridspec_kw={'height_ratios': [1.75, 1]},figsize=(9,6))

ax[0].plot(np.arange(24) + 0.5,dp2_dens,color=colormap[2],label="DP2",)
ax[0].plot(np.arange(24) + 0.5, dp1_dens, color=colormap[7], ls="-.", label="DP1")
ax[0].plot(np.arange(24) + 0.5, frey_mlt_dens, color=colormap[6], label="Frey et al. 2004")
# ax[0].hlines(.5*dp2_dens.max(),0,24,color=colormap[2])
# ax[0].hlines(.5*dp1_dens.max(),0,24,color=colormap[7])
ax[0].set_ylabel("Probability")
ax[0].set_xticks(range(24))
ax[0].set_xticklabels(bins)
ax[0].set_xlim(0,24)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

ssa_dp1 = shoredp1.values[:,1:].T.mean(axis=0)
ssa_dp2 = shoredp2.values[:,1:].T.mean(axis=0)

ax[1].plot(np.arange(24) + .5, ssa_dp1,label="DP1 Spatial Amplitude Profile",color=colormap[7],ls="-.")
ax[1].plot(np.arange(24) + .5,ssa_dp2,label="DP2 Spatial Amplitude Profile",color=colormap[2])
ax[1].hlines(0,0,24,linestyles='--',color='black',alpha=0.5,)
ax[1].set_ylabel("Amplitude")
ax[1].set_ylim(bottom=-0.5,top=.75)
ax[1].yaxis.set_ticks(np.arange(-0.5,1,0.25))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout(pad=1)

# Plotting SSA DP1 and DP2
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5, ssa_dp1,label="DP1 Spatial Amplitude Profile",color=colormap[7],ls="-.")
ax.plot(np.arange(24) + 0.5,ssa_dp2,label="DP2 Spatial Amplitude Profile",color=colormap[2])
ax.hlines(0,0,24,linestyles='--',color='black',alpha=0.5,)
ax.set_xlabel("MLT")
ax.set_ylabel("Amplitude")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.set_xlim(0,24)
ax.yaxis.set_ticks(np.arange(-0.5,1,0.25))
ax.set_ylim(bottom=-0.5,top=.75)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)


# %% Plotting fitted distributions

# Plotting Isolated fit from (Isolated - Convection) and Convection
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,iso_mlt_counts,color=colormap[1],label="Isolated substorm: No. of events: {}".format(len(iso_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_iso,color=colormap[9],ls="--",label="Fitted Isolated substorm: DP1: {} and DP2: {} (GoF:{:.3f})".format(n_substorm_iso, len(iso_onsets) - n_substorm_iso,chi_iso),)
ax.plot(np.arange(24) + 0.5,n_substorm_iso * dp1_dens,color=colormap[7],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_convec_iso * dp2_dens,color=colormap[2],ls='--',label="DP2 contribution")
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_iso:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)

ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Compound fit from (Isolated - Convection) and Convection
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,comp_mlt_counts,color=colormap[3],label="Compound substorm: No. of events: {}".format(len(comp_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_comp,color=colormap[9],ls="--",label="Fitted Compound substorm: DP1: {} and DP2: {} (GoF:{:.0f})".format(n_substorm_comp, len(comp_onsets) - n_substorm_comp,chi_comp),)
ax.plot(np.arange(24) + 0.5,n_substorm_comp * dp1_dens,color=colormap[7],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_convec_comp * dp2_dens,color=colormap[2],ls='--',label="DP2 contribution",)
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_comp:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting After Convection fit from (Isolated - Convection) and Convection
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,after_convec_mlt_counts,color=colormap[4],label="After Convection substorm: No. of events: {}".format(len(after_convec_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_after_convec,color=colormap[9],ls="--",label="Fitted After Convection substorm: DP1: {} and DP2: {} (GoF:{:.0e})".format(n_substorm_after_convec, len(after_convec_onsets) - n_substorm_after_convec,chi_after_convec))
ax.plot(np.arange(24) + 0.5,(n_substorm_after_convec) * dp1_dens,color=colormap[7],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_convec_after_convec * dp2_dens,color=colormap[2],ls='--',label="DP2 contribution",)
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_after_convec:.1e}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Other fit from (Isolated - Convection) and Convection
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,other_mlt_counts,color=colormap[5],label="Other substorm: No. of events: {}".format(len(other_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_other,color=colormap[9],ls="--",label="Fitted Other substorm: DP1: {} and DP2: {} (GoF:{:.0f})".format(n_substorm_other, len(other_onsets) - n_substorm_other,chi_other),)
ax.plot(np.arange(24) + 0.5,(n_substorm_other) * dp1_dens,color=colormap[7],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_convec_other * dp2_dens,color=colormap[2],ls='--',label="DP2 contribution",)
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_other:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# %% Printing out numbers

nevents = len(expansiondf)
n_convec = len(convec_expansiondf)
n_substorms = len(substorm_expansion)
dp1 = np.sum([n_substorm_iso, n_substorm_comp, n_substorm_after_convec, n_substorm_other])
dp2 = np.sum([n_convec_iso,n_convec_comp,n_convec_after_convec,n_convec_other,len(convec_expansiondf),])

print(f"Number of events: {total_mlt_counts.sum()}")
print(f"Number of Isolated: {isolated_onsets_mlt.size} ({isolated_onsets_mlt.size/total_mlt_counts.sum()*100:.0f}%)")
print(f"Number of Compound: {compound_onsets_mlt.size} ({compound_onsets_mlt.size/total_mlt_counts.sum()*100:.0f}%)")
print(f"Number of Convection: {convec_onsets_mlt.size} ({convec_onsets_mlt.size/total_mlt_counts.sum()*100:.0f}%)")

print(f"Number of events: {nevents}")
print(f"Number of Substorms: {n_substorms} ({n_substorms/nevents*100:.1f}%)")
print(f"Number of Convection: {n_convec} ({n_convec/nevents*100:.1f}%)")
print(f"Number of Other: {len(other_onsets)} ({len(other_onsets)/nevents:.2f}%)")
print(f"Number of After Convection: {len(after_convec_onsets)} ({len(after_convec_onsets)/nevents*100:.1f}%)")
print(f"Number of DP1: {dp1} ({dp1/nevents*100:.0f}%)")
print(f"Number of DP2: {dp2} ({dp2/nevents*100:.0f}%)")
print(dp1 + dp2 == nevents)

print(n_convec_iso+n_convec_comp+n_convec_after_convec+n_convec_other)
# %%
