# %% # Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cramervonmises_2samp

# Housekeeping
sns.set_theme(context="paper",style="whitegrid",palette="colorblind")
colormap = sns.color_palette("colorblind")

# Labelling of the MLT bins
bins = list(range(12, 24)) + list(range(0, 12))
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
# sophiedf = pd.read_csv("Data/SOPHIE_EPT90_1996-2021.txt")
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

# sophiedf["Flag"] = sophiedf["Flag"].replace(4, 0)
# sophiedf["Flag"] = sophiedf["Flag"].replace([1, 2, 3, 5, 6, 7], 1)
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
comp_mlt_counts, comp_mlt_bins = np.histogram(
    compound_onsets_mlt, bins=np.arange(0, 25)
)
comp_mlt_bins = comp_mlt_bins[:-1]
comp_mlt_counts = [*comp_mlt_counts[12:], *comp_mlt_counts[:12]]
comp_mlt_counts_err = 2 * np.sqrt(comp_mlt_counts)
comp_mlt_dens = comp_mlt_counts / np.sum(comp_mlt_counts)
comp_mlt_dens_err = comp_mlt_counts_err / np.sum(comp_mlt_counts)

# Convection expansions
convec_onsets_mlt = convec_expansiondf["MLT"].to_numpy()
convec_mlt_counts, convec_mlt_bins = np.histogram(
    convec_onsets_mlt, bins=np.arange(0, 25)
)
convec_mlt_bins = convec_mlt_bins[:-1]
convec_mlt_counts = [*convec_mlt_counts[12:], *convec_mlt_counts[:12]]
convec_mlt_counts_err = 2 * np.sqrt(convec_mlt_counts)
convec_mlt_dens = convec_mlt_counts / np.sum(convec_mlt_counts)
convec_mlt_dens_err = convec_mlt_counts_err / np.sum(convec_mlt_counts)

# After convection expansions
after_convec_mlt = after_convec_onsets["MLT"].to_numpy()
after_convec_mlt_counts, after_convec_mlt_bins = np.histogram(
    after_convec_mlt, bins=np.arange(0, 25)
)
after_convec_mlt_bins = after_convec_mlt_bins[:-1]
after_convec_mlt_counts = [*after_convec_mlt_counts[12:], *after_convec_mlt_counts[:12]]
after_convec_mlt_counts_err = 2 * np.sqrt(after_convec_mlt_counts)
after_convec_mlt_dens = after_convec_mlt_counts / np.sum(after_convec_mlt_counts)
after_convec_mlt_dens_err = after_convec_mlt_counts_err / np.sum(
    after_convec_mlt_counts
)

# Other expansions
other_mlt = other_onsets["MLT"].to_numpy()
other_mlt_counts, other_mlt_bins = np.histogram(other_mlt, bins=np.arange(0, 25))
other_mlt_bins = other_mlt_bins[:-1]
other_mlt_counts = [*other_mlt_counts[12:], *other_mlt_counts[:12]]
other_mlt_counts_err = 2 * np.sqrt(other_mlt_counts)
other_mlt_dens = other_mlt_counts / np.sum(other_mlt_counts)
other_mlt_dens_err = other_mlt_counts_err / np.sum(other_mlt_counts)

# Total of Isolated, Compound and Convection
total_mlt_counts = (
    np.array(iso_mlt_counts) + np.array(comp_mlt_counts) + np.array(convec_mlt_counts)
)
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

ax.plot(
    np.arange(24) + 0.5,
    onsets_mlt_counts,
    label="All SOPHIE Events: No. of onsets: {}".format(len(onsets_mlt)),
    alpha = 0.75
)
ax.plot(
    np.arange(24) + 0.5,
    substorm_onsets_mlt_counts,
    label="SOPHIE Substorms: No. of onsets: {}".format(len(substorm_onsets_mlt)),
    alpha = 0.75
)
ax.plot(
    np.arange(24) + 0.5,
    newell_mlt_counts,
    label="Newell & Gjerloev 2011: No. of onsets: {}".format(len(newelldf)),
    ls="-.",
)
ax.plot(
    np.arange(24) + 0.5,
    ohtani_mlt_counts,
    label="Ohtani & Gjerloev 2020: No. of onsets: {}".format(len(ohtanidf)),
    ls="-.",
)
ax.plot(
    np.arange(24) + 0.5,
    frey_mlt_counts,
    label="Frey et al. 2004: No. of onsets: {}".format(len(freydf)),
    ls="-.",
)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

# %% Fitting MLT distributions

# Fitting Isolated 3-18 MLT from Convection distribution
iso_mlt_counts_3_18 = [*iso_mlt_counts[:7], *iso_mlt_counts[15:]]
iso_mlt_counts_3_18_err = [*iso_mlt_counts_err[:7], *iso_mlt_counts_err[15:]]
chi_hist = np.inf
dist = np.zeros(24)
iso_convec_fit = np.zeros(24)
wght_hist = 0

n = 0
while (np.array(iso_mlt_counts) - dist).min() > 0:
    dist = n * np.array(convec_mlt_dens)
    dist = np.round(dist, 0)
    dist_3_18 = [*dist[:7], *dist[15:]]
    chi_sq = chi_squared_test(iso_mlt_counts_3_18, dist_3_18, iso_mlt_counts_3_18_err)
    if chi_sq < chi_hist:
        chi_hist = chi_sq
        wght_hist = n
        iso_convec_fit = dist
    n += 1

iso_minus_convec = np.array(iso_mlt_counts) - iso_convec_fit
iso_minus_convec_dens = iso_minus_convec / np.sum(iso_minus_convec)

mask = np.zeros(np.shape(iso_minus_convec))
mask[7:17] = 1
dp1 = np.where(mask, iso_minus_convec, 0)
dp1_dens = dp1 / np.sum(dp1)

mask = np.zeros(np.shape(iso_minus_convec))
mask[:7] = 1
mask[17:] = 1
dp2 = np.where(mask, iso_minus_convec, 0) + convec_mlt_counts
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

# Plotting Histograms (Isolated, Compound, Convection Expansion, Total)
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    iso_mlt_counts,
    color=colormap[1],
    label="Isolated: No. of onsets: {}".format(len(iso_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    comp_mlt_counts,
    color=colormap[3],
    label="Compound: No. of onsets: {}".format(len(comp_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    convec_mlt_counts,
    color=colormap[2],
    label="Convection: No. of onsets: {}".format(len(convec_expansiondf)),
)
ax.plot(
    np.arange(24) + 0.5,
    total_mlt_counts,
    color=colormap[0],
    label="Total: No. of onsets: {}".format(np.sum(total_mlt_counts)),
)
ax.plot(
    np.arange(24) + 0.5,
    frey_mlt_counts,
    color=colormap[6],
    label="Frey et al. 2004: No. of onsets: {}".format(len(freydf)),
)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)


# Plotting Histogram (Isolated, Convection Fit, Residual)
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5, iso_mlt_counts, color=colormap[1], label="Isolated Substorms"
)
# ax.plot(convec_mlt_counts,color=colormap[3],label='Convection Expansion Phase: No. of onsets: {}'.format(len(convec_expansiondf)))
ax.plot(
    np.arange(24) + 0.5,
    iso_convec_fit,
    color=colormap[8],
    ls="-.",
    label="Scaled Convection Fit",
)
ax.plot(
    np.arange(24) + 0.5,
    iso_minus_convec,
    color=colormap[7],
    ls="-.",
    label="DP1 Perturbation = Isolated Substorms - Scaled Convection Fit",
)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Probability Distributions (Residual, Convection, Frey)
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    dp2_dens,
    color=colormap[2],
    label="Convection (DP2 Perturbation)",
)
ax.plot(
    np.arange(24) + 0.5, dp1_dens, color=colormap[7], ls="-.", label="DP1 Perturbation"
)
ax.plot(np.arange(24) + 0.5, frey_mlt_dens, color=colormap[6], label="Frey et al. 2004")
ax.set_xlabel("MLT")
ax.set_ylabel("Probability")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(
    loc="upper left",
)

fig.tight_layout(pad=1)

# %% Plotting fitted distributions

# Plotting Isolated fit from (Isolated - Convection) and Convection

cvm_iso = cramervonmises_2samp(iso_mlt_counts, fit_iso).statistic
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    iso_mlt_counts,
    color=colormap[1],
    label="Isolated: No. of onsets: {}".format(len(iso_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    fit_iso,
    color=colormap[9],
    ls="--",
    label="Fitted Isolated: DP1: {} and DP2: {}".format(
        n_substorm_iso, len(iso_onsets) - n_substorm_iso
    ),
)
ax.plot(
    np.arange(24) + 0.5,
    n_substorm_iso * dp1_dens,
    color=colormap[7],
    ls="-.",
    label="DP1 contribution",
)
ax.plot(
    np.arange(24) + 0.5,
    n_convec_iso * dp2_dens,
    color=colormap[8],
    label="DP2 contribution"
   
)
ax.text(s=f"Cramer-von Mises Goodness of Fit: {cvm_iso:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)

ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Compound fit from (Isolated - Convection) and Convection

cvm_comp = cramervonmises_2samp(comp_mlt_counts, fit_comp).statistic
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    comp_mlt_counts,
    color=colormap[3],
    label="Compound: No. of onsets: {}".format(len(comp_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    fit_comp,
    color=colormap[9],
    ls="--",
    label="Fitted Compound: DP1: {} and DP2: {}".format(
        n_substorm_comp, len(comp_onsets) - n_substorm_comp
    ),
)
ax.plot(
    np.arange(24) + 0.5,
    n_substorm_comp * dp1_dens,
    color=colormap[7],
    ls="-.",
    label="DP1 contribution",
)
ax.plot(
    np.arange(24) + 0.5,
    n_convec_comp * dp2_dens,
    color=colormap[8],
    label="DP2 contribution",
)
ax.text(s=f"Cramer-von Mises Goodness of Fit: {cvm_comp:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting After Convection fit from (Isolated - Convection) and Convection

cvm_after_convec = cramervonmises_2samp(after_convec_mlt_counts, fit_after_convec).statistic
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    after_convec_mlt_counts,
    color=colormap[4],
    label="After Convection Onsets: No. of onsets: {}".format(len(after_convec_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    fit_after_convec,
    color=colormap[9],
    ls="--",
    label="Fitted After Convection Onsets: DP1: {} and DP2: {}".format(
        n_substorm_after_convec, len(after_convec_onsets) - n_substorm_after_convec
    ),
)
ax.plot(
    np.arange(24) + 0.5,
    (n_substorm_after_convec) * dp1_dens,
    color=colormap[7],
    ls="-.",
    label="DP1 contribution",
)
ax.plot(
    np.arange(24) + 0.5,
    n_convec_after_convec * dp2_dens,
    color=colormap[8],
    label="DP2 contribution",
)
ax.text(s=f"Cramer-von Mises Goodness of Fit: {cvm_after_convec:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Plotting Other fit from (Isolated - Convection) and Convection

cvm_other = cramervonmises_2samp(other_mlt_counts, fit_other).statistic
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(24) + 0.5,
    other_mlt_counts,
    color=colormap[5],
    label="Other Expansion Phases: No. of onsets: {}".format(len(other_onsets)),
)
ax.plot(
    np.arange(24) + 0.5,
    fit_other,
    color=colormap[9],
    ls="--",
    label="Fitted other: DP1: {} and DP2: {}".format(
        n_substorm_other, len(other_onsets) - n_substorm_other
    ),
)
ax.plot(
    np.arange(24) + 0.5,
    (n_substorm_other) * dp1_dens,
    color=colormap[7],
    ls="-.",
    label="DP1 contribution",
)
ax.plot(
    np.arange(24) + 0.5,
    n_convec_other * dp2_dens,
    color=colormap[8],
    label="DP2 contribution",
)
ax.text(s=f"Cramer-von Mises Goodness of Fit: {cvm_other:.3f}",x=0.025,y=0.95,transform=ax.transAxes,)
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)


# %% Printing out numbers

nevents = len(expansiondf)
dp1 = np.sum([n_substorm_iso, n_substorm_comp, n_substorm_after_convec, n_substorm_other])
dp2 = np.sum(
    [
        n_convec_iso,
        n_convec_comp,
        n_convec_after_convec,
        n_convec_other,
        len(convec_expansiondf),
    ]
)

print(f"Number of events: {nevents}")
print(f"Number of DP1: {dp1} ({dp1/nevents:.2f}%)")
print(f"Number of DP2: {dp2} ({dp2/nevents:.2f}%)")
print(dp1 + dp2 == nevents)

# %%
