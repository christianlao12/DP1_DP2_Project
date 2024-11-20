# %% # Importing Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Housekeeping
sns.set_theme(context="paper",style="whitegrid",palette="colorblind")
colormap = sns.color_palette("colorblind")

# Labelling of the MLT bins
bins = np.roll(np.arange(0, 24), 12)
bins = list(map(str, bins))

# Function to calculate chi squared test
def chi_squared_test(measured, model, uncertainty):
    return np.sum(
        np.nan_to_num(
            (np.array(measured) - np.array(model)) ** 2 / np.array(uncertainty) ** 2
        )
    )

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
shoredf = pd.read_csv("Data/ShoreData/ShoreSpatialAmp.csv")

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

# Shore MLT
shoredp1 = shoredf['DP1'].copy().to_numpy()
shoredp2 = shoredf['DP2'].copy().to_numpy()

shoredp1_dens = shoredp1 / np.sum(shoredp1)
shoredp2_dens = shoredp2 / np.sum(shoredp2)


# %% Fitting MLT distributions

# Fitting to Substorm distribution by weighted sum of DP1 and DP2
chi_substorm = np.inf
fit_substorm = []
n_dp1_substorm = 0

for i in range(len(substorm_onsets_mlt)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(substorm_onsets_mlt) - i)
    chi_sq = chi_squared_test(substorm_onsets_mlt_counts, dist, substorm_onsets_mlt_counts_err)
    if chi_sq < chi_substorm:
        chi_substorm = chi_sq
        n_dp1_substorm = i
        fit_substorm = dist

fit_substorm_dens = fit_substorm / np.sum(fit_substorm)
n_dp2_substorm = len(substorm_onsets_mlt) - n_dp1_substorm

# Fitting to Isolated distribution by weighted sum of DP1 and DP2
chi_iso = np.inf
fit_iso = []
n_dp1_iso = 0

for i in range(len(iso_onsets)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(iso_onsets) - i)
    chi_sq = chi_squared_test(iso_mlt_counts, dist, iso_mlt_counts_err)
    if chi_sq < chi_iso:
        chi_iso = chi_sq
        n_dp1_iso = i
        fit_iso = dist

fit_iso_dens = fit_iso / np.sum(fit_iso)
n_dp2_iso = len(iso_onsets) - n_dp1_iso

# Fitting to Compound distribution by weighted sum of DP1 and DP2
chi_comp = np.inf
fit_comp = []
n_dp1_comp = 0

for i in range(len(comp_onsets)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(comp_onsets) - i)
    chi_sq = chi_squared_test(comp_mlt_counts, dist, comp_mlt_counts_err)
    if chi_sq < chi_comp:
        chi_comp = chi_sq
        n_dp1_comp = i
        fit_comp = dist

fit_comp_dens = fit_comp / np.sum(fit_comp)
n_dp2_comp = len(comp_onsets) - n_dp1_comp

# Fitting to After Convection distribution by weighted sum of DP1 and DP2
chi_after_convec = np.inf
fit_after_convec = []
n_dp1_after_convec = 0

for i in range(len(after_convec_onsets)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(after_convec_onsets) - i)
    chi_sq = chi_squared_test(after_convec_mlt_counts, dist, after_convec_mlt_counts_err)
    if chi_sq < chi_after_convec:
        chi_after_convec = chi_sq
        n_dp1_after_convec = i
        fit_after_convec = dist

fit_after_convec_dens = fit_after_convec / np.sum(fit_after_convec)
n_dp2_after_convec = len(after_convec_onsets) - n_dp1_after_convec

# Fitting to Other distribution by weighted sum of DP1 and DP2
chi_other = np.inf
fit_other = []
n_dp1_other = 0

for i in range(len(other_onsets)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(other_onsets) - i)
    chi_sq = chi_squared_test(other_mlt_counts, dist, other_mlt_counts_err)
    if chi_sq < chi_other:
        chi_other = chi_sq
        n_dp1_other = i
        fit_other = dist

fit_other_dens = fit_other / np.sum(fit_other)
n_dp2_other = len(other_onsets) - n_dp1_other

# Fitting to Convection distribution by weighted sum of DP1 and DP2
chi_convec = np.inf
fit_convec = []
n_dp1_convec = 0

for i in range(len(convec_expansiondf)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(convec_expansiondf) - i)
    chi_sq = chi_squared_test(convec_mlt_counts, dist, convec_mlt_counts_err)
    if chi_sq < chi_convec:
        chi_convec = chi_sq
        n_dp1_convec = i
        fit_convec = dist

fit_convec_dens = fit_convec / np.sum(fit_convec)
n_dp2_convec = len(convec_expansiondf) - n_dp1_convec

# Fitting to Newell distribution by weighted sum of DP1 and DP2
chi_newell = np.inf
fit_newell = []
n_dp1_newell = 0

for i in range(len(newell_mlt)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(newell_mlt) - i)
    chi_sq = chi_squared_test(newell_mlt_counts, dist, newell_mlt_counts_err)
    if chi_sq < chi_newell:
        chi_newell = chi_sq
        n_dp1_newell = i
        fit_newell = dist

fit_newell_dens = fit_newell / np.sum(fit_newell)
n_dp2_newell = len(newell_mlt) - n_dp1_newell

# Fitting to Ohtani distribution by weighted sum of DP1 and DP2
chi_ohtani = np.inf
fit_ohtani = []
n_dp1_ohtani = 0

for i in range(len(ohtani_mlt)):
    dist = np.array(shoredp1_dens) * i + np.array(shoredp2_dens) * (len(ohtani_mlt) - i)
    dist_test = dist.copy()
    dist_test[np.where(np.array(ohtani_mlt_counts)==0)] = 0
    chi_sq = chi_squared_test(ohtani_mlt_counts, dist_test, ohtani_mlt_counts_err)
    if chi_sq < chi_ohtani:
        chi_ohtani = chi_sq
        n_dp1_ohtani = i
        fit_ohtani = dist

fit_ohtani_dens = fit_ohtani / np.sum(fit_ohtani)
n_dp2_ohtani = len(ohtani_mlt) - n_dp1_ohtani

# %% Plotting Known distributions

# Plotting Probability Distributions (Substorm, Convection, Frey)
fig, ax = plt.subplots(dpi=300)

ax.plot(np.arange(24) + 0.5,substorm_onsets_mlt_dens,color=colormap[0],label="Substorm: No. of onsets: {}".format(len(substorm_onsets_mlt)),)
ax.plot(np.arange(24) + 0.5,convec_mlt_dens,color=colormap[1],label="Convection Interval: No. of onsets: {}".format(len(convec_expansiondf)),)
ax.set_xlabel("MLT")
ax.set_ylabel("Probability")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Amplitude Distributions DP1 and DP2
fig, ax = plt.subplots(dpi=300, figsize=(8, 4))

ax.plot(np.arange(24) + 0.5,shoredp1,color=colormap[3],ls="-.",label="DP1",)
ax.plot(np.arange(24) + 0.5,shoredp2,color=colormap[2],label="DP2",)
ax.set_xlabel("MLT")
ax.set_ylabel("Amplitude")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="upper right",)

fig.tight_layout(pad=1)

# Plotting Normalised Amplitude Distributions DP1 and DP2
fig, ax = plt.subplots(dpi=300, figsize=(8, 4))

ax.plot(np.arange(24) + 0.5,shoredp1_dens,color=colormap[3],ls="-.",label="DP1",)
ax.plot(np.arange(24) + 0.5,shoredp2_dens,color=colormap[2],label="DP2",)
ax.set_xlabel("MLT")
ax.set_ylabel("Amplitude")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="upper right",)

fig.tight_layout(pad=1)


# %% Plotting fitted distributions

# Plotting Substorm fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,substorm_onsets_mlt_counts,color=colormap[0],label="Substorm: No. of onsets: {}".format(len(substorm_onsets_mlt)),)
ax.plot(np.arange(24) + 0.5,fit_substorm,color=colormap[9],ls="--",label="Fitted Substorm: DP1: {} and DP2: {}".format(n_dp1_substorm, len(substorm_onsets_mlt) - n_dp1_substorm),)
ax.plot(np.arange(24) + 0.5,n_dp1_substorm * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_substorm * shoredp2_dens,color=colormap[2],label="DP2 contribution",)
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_substorm:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Convection fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,convec_mlt_counts,color=colormap[1],label="Convection: No. of onsets: {}".format(len(convec_expansiondf)))
ax.plot(np.arange(24) + 0.5,fit_convec,color=colormap[9],ls="--",label="Fitted Convection: DP1: {} and DP2: {}".format(n_dp1_convec, len(convec_expansiondf) - n_dp1_convec))
ax.plot(np.arange(24) + 0.5,(n_dp1_convec) * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_convec * shoredp2_dens,color=colormap[2],label="DP2 contribution",)
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_convec:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Isolated fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,iso_mlt_counts,color=colormap[4],label="Isolated: No. of onsets: {}".format(len(iso_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_iso,color=colormap[9],ls="--",label="Fitted Isolated: DP1: {} and DP2: {}".format(n_dp1_iso, len(iso_onsets) - n_dp1_iso),)
ax.plot(np.arange(24) + 0.5,n_dp1_iso * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_iso * shoredp2_dens,color=colormap[2],label="DP2 contribution")
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_iso:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Compound fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,comp_mlt_counts,color=colormap[5],label="Compound: No. of onsets: {}".format(len(comp_onsets)),)
ax.plot(np.arange(24) + 0.5,fit_comp,color=colormap[9],ls="--",label="Fitted Compound: DP1: {} and DP2: {}".format(n_dp1_comp, len(comp_onsets) - n_dp1_comp),)
ax.plot(np.arange(24) + 0.5,n_dp1_comp * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_comp * shoredp2_dens,color=colormap[2],label="DP2 contribution")
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_comp:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Newell fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,newell_mlt_counts,color=colormap[6],label="Newell et al. 2011: No. of onsets: {}".format(len(newell_mlt)),)
ax.plot(np.arange(24) + 0.5,fit_newell,color=colormap[9],ls="--",label="Fitted Newell: DP1: {} and DP2: {}".format(n_dp1_newell, len(newell_mlt) - n_dp1_newell),)
ax.plot(np.arange(24) + 0.5,n_dp1_newell * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_newell * shoredp2_dens,color=colormap[2],label="DP2 contribution")
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_newell:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Ohtani fit from DP1 and DP2

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))

ax.plot(np.arange(24) + 0.5,ohtani_mlt_counts,color=colormap[7],label="Ohtani et al. 2020: No. of onsets: {}".format(len(ohtani_mlt)),)
ax.plot(np.arange(24) + 0.5,fit_ohtani,color=colormap[9],ls="--",label="Fitted Ohtani: DP1: {} and DP2: {}".format(n_dp1_ohtani, len(ohtani_mlt) - n_dp1_ohtani),)
ax.plot(np.arange(24) + 0.5,n_dp1_ohtani * shoredp1_dens,color=colormap[3],ls="-.",label="DP1 contribution",)
ax.plot(np.arange(24) + 0.5,n_dp2_ohtani * shoredp2_dens,color=colormap[2],label="DP2 contribution")
# ax.text(s=f"Chi-squared Goodness of Fit: {chi_ohtani:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)


# %% Printing out numbers

nevents = len(expansiondf)
n_dp2 = len(convec_expansiondf)
n_dp1s = len(substorm_expansion)
dp1 = np.sum([n_dp1_iso, n_dp1_comp, n_dp1_after_convec, n_dp1_other,n_dp1_convec])
dp2 = np.sum([n_dp2_iso,n_dp2_comp,n_dp2_after_convec,n_dp2_other,n_dp2_convec])

dp1_2 = np.sum([n_dp1_substorm, n_dp1_convec])
dp2_2 = np.sum([n_dp2_substorm, n_dp2_convec])


print(f"Number of events: {nevents}")
print(f"Number of Substorms: {n_dp1s} ({n_dp1s/nevents:.2f}%)")
print(f"Number of Convection: {n_dp2} ({n_dp2/nevents:.2f}%)")
print(f"Number of DP1: {dp1} ({dp1/nevents:.2f}%)")
print(f"Number of DP2: {dp2} ({dp2/nevents:.2f}%)")
print(dp1 + dp2 == nevents)

print(f"Number of DP1: {dp1_2} ({dp1_2/nevents:.2f}%)")
print(f"Number of DP2: {dp2_2} ({dp2_2/nevents:.2f}%)")
print(dp1_2 + dp2_2 == nevents)

print(f"Substorm: DP1: {n_dp1_substorm} ({n_dp1_substorm/substorm_onsets_mlt.size:.2f}%) DP2: {n_dp2_substorm} ({n_dp2_substorm/substorm_onsets_mlt.size:.2f}%)")
print(f"Convection: DP1: {n_dp1_convec} ({n_dp1_convec/convec_onsets_mlt.size:.2f}%) DP2: {n_dp2_convec} ({n_dp2_convec/convec_onsets_mlt.size:.2f}%)")
print(f"Isolated: DP1: {n_dp1_iso} ({n_dp1_iso/isolated_onsets_mlt.size:.2f}%) DP2: {n_dp2_iso} ({n_dp2_iso/isolated_onsets_mlt.size:.2f}%)")
print(f"Compound: DP1: {n_dp1_comp} ({n_dp1_comp/compound_onsets_mlt.size:.2f}%) DP2: {n_dp2_comp} ({n_dp2_comp/compound_onsets_mlt.size:.2f}%)")
# %%