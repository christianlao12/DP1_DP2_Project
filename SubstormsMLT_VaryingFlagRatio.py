# %%
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook",style="whitegrid",palette="colorblind",)
colormap = sns.color_palette("colorblind", as_cmap=True)

# Function to calculate chi squared test
def chi_squared_test(measured, model, uncertainty):
    return np.sum(np.nan_to_num((np.array(measured) - np.array(model)) ** 2 / np.array(uncertainty) ** 2))


# %% Loading in SOPHIE Data
# Loading in SOPHIE SMU Threshold 1
sophiedf_smu1 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_01.csv",)
sophiedf_smu1["Date_UTC"] = pd.to_datetime(sophiedf_smu1["Date_UTC"])
sophiedf_smu1["Duration"] = np.append(np.diff(sophiedf_smu1["Date_UTC"].to_numpy()), 0)
sophiedf_smu1["Delbay"] = pd.to_numeric(sophiedf_smu1["DeltaSML"], errors="coerce")
sophiedf_smu1.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu1 = sophiedf_smu1[sophiedf_smu1["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 2
sophiedf_smu2 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_02.csv",)
sophiedf_smu2["Date_UTC"] = pd.to_datetime(sophiedf_smu2["Date_UTC"])
sophiedf_smu2["Duration"] = np.append(np.diff(sophiedf_smu2["Date_UTC"].to_numpy()), 0)
sophiedf_smu2["Delbay"] = pd.to_numeric(sophiedf_smu2["DeltaSML"], errors="coerce")
sophiedf_smu2.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu2 = sophiedf_smu2[sophiedf_smu2["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 3
sophiedf_smu3 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_03.csv")
sophiedf_smu3["Date_UTC"] = pd.to_datetime(sophiedf_smu3["Date_UTC"])
sophiedf_smu3["Duration"] = np.append(np.diff(sophiedf_smu3["Date_UTC"].to_numpy()), 0)
sophiedf_smu3["Delbay"] = pd.to_numeric(sophiedf_smu3["DeltaSML"], errors="coerce")
sophiedf_smu3.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu3 = sophiedf_smu3[sophiedf_smu3["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 4
sophiedf_smu4 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_04.csv",)
sophiedf_smu4["Date_UTC"] = pd.to_datetime(sophiedf_smu4["Date_UTC"])
sophiedf_smu4["Duration"] = np.append(np.diff(sophiedf_smu4["Date_UTC"].to_numpy()), 0)
sophiedf_smu4["Delbay"] = pd.to_numeric(sophiedf_smu4["DeltaSML"], errors="coerce")
sophiedf_smu4.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu4 = sophiedf_smu4[sophiedf_smu4["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 5
sophiedf_smu5 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_05.csv",)
sophiedf_smu5["Date_UTC"] = pd.to_datetime(sophiedf_smu5["Date_UTC"])
sophiedf_smu5["Duration"] = np.append(np.diff(sophiedf_smu5["Date_UTC"].to_numpy()), 0)
sophiedf_smu5["Delbay"] = pd.to_numeric(sophiedf_smu5["DeltaSML"], errors="coerce")
sophiedf_smu5.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu5 = sophiedf_smu5[sophiedf_smu5["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 6
sophiedf_smu6 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_06.csv",)
sophiedf_smu6["Date_UTC"] = pd.to_datetime(sophiedf_smu6["Date_UTC"])
sophiedf_smu6["Duration"] = np.append(np.diff(sophiedf_smu6["Date_UTC"].to_numpy()), 0)
sophiedf_smu6["Delbay"] = pd.to_numeric(sophiedf_smu6["DeltaSML"], errors="coerce")
sophiedf_smu6.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu6 = sophiedf_smu6[sophiedf_smu6["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 7
sophiedf_smu7 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_07.csv",)
sophiedf_smu7["Date_UTC"] = pd.to_datetime(sophiedf_smu7["Date_UTC"])
sophiedf_smu7["Duration"] = np.append(np.diff(sophiedf_smu7["Date_UTC"].to_numpy()), 0)
sophiedf_smu7["Delbay"] = pd.to_numeric(sophiedf_smu7["DeltaSML"], errors="coerce")
sophiedf_smu7.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu7 = sophiedf_smu7[sophiedf_smu7["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 8
sophiedf_smu8 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_08.csv",)
sophiedf_smu8["Date_UTC"] = pd.to_datetime(sophiedf_smu8["Date_UTC"])
sophiedf_smu8["Duration"] = np.append(np.diff(sophiedf_smu8["Date_UTC"].to_numpy()), 0)
sophiedf_smu8["Delbay"] = pd.to_numeric(sophiedf_smu8["DeltaSML"], errors="coerce")
sophiedf_smu8.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu8 = sophiedf_smu8[sophiedf_smu8["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 9
sophiedf_smu9 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_09.csv",)
sophiedf_smu9["Date_UTC"] = pd.to_datetime(sophiedf_smu9["Date_UTC"])
sophiedf_smu9["Duration"] = np.append(np.diff(sophiedf_smu9["Date_UTC"].to_numpy()), 0)
sophiedf_smu9["Delbay"] = pd.to_numeric(sophiedf_smu9["DeltaSML"], errors="coerce")
sophiedf_smu9.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu9 = sophiedf_smu9[sophiedf_smu9["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# Loading in SOPHIE SMU Threshold 10
sophiedf_smu10 = pd.read_csv("Data/SOPHIE_VaryingFlagv2/EPT90_Threshold_10.csv",)
sophiedf_smu10["Date_UTC"] = pd.to_datetime(sophiedf_smu10["Date_UTC"])
sophiedf_smu10["Duration"] = np.append(np.diff(sophiedf_smu10["Date_UTC"].to_numpy()), 0)
sophiedf_smu10["Delbay"] = pd.to_numeric(sophiedf_smu10["DeltaSML"], errors="coerce")
sophiedf_smu10.drop(columns=["DeltaSML"], inplace=True)
sophiedf_smu10 = sophiedf_smu10[sophiedf_smu10["Date_UTC"].between("1997", "2020", inclusive='left')].reset_index(drop=True)

# %% SOPHIE Phases
# Onset types

# Function to process SOPHIE data
def process_sophiedf(sophiedf):
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
    sophiedf["Isolated Onset"] = iso_arr

    # Compound Onsets
    comp_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
    for i in range(2, len(sophiedf["Date_UTC"]) - 1):
        if (
            (sophiedf.iloc[i - 2]["Phase"] ==2)
            and (sophiedf.iloc[i - 1]["Phase"] == 3)
            and (sophiedf.iloc[i]["Phase"] == 2)
        ):
            comp_arr[i] = 1
    for i, val in enumerate(comp_arr):
        if val == 1:
            comp_arr[i - 2] = 1
    sophiedf["Compound Onset"] = comp_arr

    # Excluding onsets after a convection interval
    newflag_arr = sophiedf["Flag"].to_numpy().copy()
    for i in range(1, len(sophiedf["Flag"])):
        if newflag_arr[i] == 1 or (
            newflag_arr[i - 1] == 1 and sophiedf.iloc[i]["Phase"] != 1
        ):
            newflag_arr[i] = 1  
    sophiedf["NewFlag"] = newflag_arr

    return sophiedf


# Use the function
sophiedf_smu1 = process_sophiedf(sophiedf_smu1)
sophiedf_smu2 = process_sophiedf(sophiedf_smu2)
sophiedf_smu3 = process_sophiedf(sophiedf_smu3)
sophiedf_smu4 = process_sophiedf(sophiedf_smu4)
sophiedf_smu5 = process_sophiedf(sophiedf_smu5)
sophiedf_smu6 = process_sophiedf(sophiedf_smu6)
sophiedf_smu7 = process_sophiedf(sophiedf_smu7)
sophiedf_smu8 = process_sophiedf(sophiedf_smu8)
sophiedf_smu9 = process_sophiedf(sophiedf_smu9)
sophiedf_smu10 = process_sophiedf(sophiedf_smu10)


# %% Types for each threshold
# Calculating types for each threshold
# Functions
def chain_calculation(df):
    expansions = df.iloc[np.where(df["Phase"] == 2)].reset_index(drop=True)
    convec_expansions = expansions.iloc[np.where(expansions["Flag"] == 1)]
    substorms = np.where(expansions["Flag"] == 0)[0]
    isolated = np.intersect1d(
        np.where(expansions["Isolated Onset"] == 1),
        np.where(expansions["NewFlag"] == 0),
    )
    iso_onsets = expansions.iloc[isolated]
    compound = np.intersect1d(
        np.where(expansions["Compound Onset"] == 1),
        np.where(expansions["NewFlag"] == 0),
    )
    comp_onsets = expansions.iloc[compound]
    after_convec = np.intersect1d(
        np.where(expansions["Phase"] == 2),
        np.setdiff1d(
            np.where(expansions["NewFlag"] == 1)[0],
            np.where(expansions["Flag"] == 1)[0],
        ),
    )
    onsets_after_convec = expansions.iloc[after_convec]
    geg = np.setdiff1d(substorms,np.concatenate((isolated,compound,after_convec)))
    gegdf = expansions.iloc[geg]
    return (
        expansions,
        convec_expansions,
        iso_onsets,
        comp_onsets,
        onsets_after_convec,
        gegdf,
    )


# SOPHIE SMU Threshold 1
(
    expansiondf_smu_1,
    convec_expansiondf_smu_1,
    iso_onsets_smu_1,
    comp_onsets_smu_1,
    onsets_after_convec_smu_1,
    gegdf_smu_1,
) = chain_calculation(sophiedf_smu1)

# SOPHIE SMU Threshold 2
(
    expansiondf_smu_2,
    convec_expansiondf_smu_2,
    iso_onsets_smu_2,
    comp_onsets_smu_2,
    onsets_after_convec_smu_2,
    gegdf_smu_2,
) = chain_calculation(sophiedf_smu2)

# SOPHIE SMU Threshold 3
(
    expansiondf_smu_3,
    convec_expansiondf_smu_3,
    iso_onsets_smu_3,
    comp_onsets_smu_3,
    onsets_after_convec_smu_3,
    gegdf_smu_3,
) = chain_calculation(sophiedf_smu3)

# SOPHIE SMU Threshold 4
(
    expansiondf_smu_4,
    convec_expansiondf_smu_4,
    iso_onsets_smu_4,
    comp_onsets_smu_4,
    onsets_after_convec_smu_4,
    gegdf_smu_4,
) = chain_calculation(sophiedf_smu4)

# SOPHIE SMU Threshold 5
(
    expansiondf_smu_5,
    convec_expansiondf_smu_5,
    iso_onsets_smu_5,
    comp_onsets_smu_5,
    onsets_after_convec_smu_5,
    gegdf_smu_5,
) = chain_calculation(sophiedf_smu5)

# SOPHIE SMU Threshold 6
(
    expansiondf_smu_6,
    convec_expansiondf_smu_6,
    iso_onsets_smu_6,
    comp_onsets_smu_6,
    onsets_after_convec_smu_6,
    gegdf_smu_6,
) = chain_calculation(sophiedf_smu6)

# SOPHIE SMU Threshold 7
(
    expansiondf_smu_7,
    convec_expansiondf_smu_7,
    iso_onsets_smu_7,
    comp_onsets_smu_7,
    onsets_after_convec_smu_7,
    gegdf_smu_7,
) = chain_calculation(sophiedf_smu7)

# SOPHIE SMU Threshold 8
(
    expansiondf_smu_8,
    convec_expansiondf_smu_8,
    iso_onsets_smu_8,
    comp_onsets_smu_8,
    onsets_after_convec_smu_8,
    gegdf_smu_8,
) = chain_calculation(sophiedf_smu8)

# SOPHIE SMU Threshold 9
(
    expansiondf_smu_9,
    convec_expansiondf_smu_9,
    iso_onsets_smu_9,
    comp_onsets_smu_9,
    onsets_after_convec_smu_9,
    gegdf_smu_9,
) = chain_calculation(sophiedf_smu9)

# SOPHIE SMU Threshold 10
(
    expansiondf_smu_10,
    convec_expansiondf_smu_10,
    iso_onsets_smu_10,
    comp_onsets_smu_10,
    onsets_after_convec_smu_10,
    gegdf_smu_10,
) = chain_calculation(sophiedf_smu10)


df = pd.DataFrame(
    {
        "1": [
            len(expansiondf_smu_1),
            len(iso_onsets_smu_1),
            len(convec_expansiondf_smu_1),
            len(comp_onsets_smu_1),
            len(onsets_after_convec_smu_1),
            len(gegdf_smu_1),
        ],
        "2": [
            len(expansiondf_smu_2),
            len(iso_onsets_smu_2),
            len(convec_expansiondf_smu_2),
            len(comp_onsets_smu_2),
            len(onsets_after_convec_smu_2),
            len(gegdf_smu_2),
        ],
        "3": [
            len(expansiondf_smu_3),
            len(iso_onsets_smu_3),
            len(convec_expansiondf_smu_3),
            len(comp_onsets_smu_3),
            len(onsets_after_convec_smu_3),
            len(gegdf_smu_3),
        ],
        "4": [
            len(expansiondf_smu_4),
            len(iso_onsets_smu_4),
            len(convec_expansiondf_smu_4),
            len(comp_onsets_smu_4),
            len(onsets_after_convec_smu_4),
            len(gegdf_smu_4),
        ],
        "5": [
            len(expansiondf_smu_5),
            len(iso_onsets_smu_5),
            len(convec_expansiondf_smu_5),
            len(comp_onsets_smu_5),
            len(onsets_after_convec_smu_5),
            len(gegdf_smu_5),
        ],
        "6": [
            len(expansiondf_smu_6),
            len(iso_onsets_smu_6),
            len(convec_expansiondf_smu_6),
            len(comp_onsets_smu_6),
            len(onsets_after_convec_smu_6),
            len(gegdf_smu_6),
        ],
        "7": [
            len(expansiondf_smu_7),
            len(iso_onsets_smu_7),
            len(convec_expansiondf_smu_7),
            len(comp_onsets_smu_7),
            len(onsets_after_convec_smu_7),
            len(gegdf_smu_7),
        ],
        "8": [
            len(expansiondf_smu_8),
            len(iso_onsets_smu_8),
            len(convec_expansiondf_smu_8),
            len(comp_onsets_smu_8),
            len(onsets_after_convec_smu_8),
            len(gegdf_smu_8),
        ],
        "9": [
            len(expansiondf_smu_9),
            len(iso_onsets_smu_9),
            len(convec_expansiondf_smu_9),
            len(comp_onsets_smu_9),
            len(onsets_after_convec_smu_9),
            len(gegdf_smu_9),
        ],
        "10": [
            len(expansiondf_smu_10),
            len(iso_onsets_smu_10),
            len(convec_expansiondf_smu_10),
            len(comp_onsets_smu_10),
            len(onsets_after_convec_smu_10),
            len(gegdf_smu_10),
        ],
    },
    index=[
        "All Onsets",
        "Isolated",
        "Convection",
        "Compound",
        "After Convection",
        "Other",
    ],
)

df.to_csv("Outputs/OnsetTypes.csv")
df.drop("All Onsets", inplace=True)

fig, ax = plt.subplots(dpi=300)

df.plot(ax=ax, kind="bar", stacked=False, figsize=(10, 5))
ax.set_xlabel("Onset Type")
ax.set_ylabel("Counts")
ax.legend(title="Threshold")

# df.transpose().plot(kind="bar", stacked=False, figsize=(10, 5))


# %%# Substorm MLT Distribution Functions

# Distribution function
def mlt_distribution(df):
    onsets = df["MLT"].to_numpy()

    onsets_counts, onsets_bins = np.histogram(onsets, bins=np.arange(0, 25))
    onsets_bins = onsets_bins[:-1]
    onsets_counts = [*onsets_counts[12:], *onsets_counts[:12]]
    onsets_counts_err = 2 * np.sqrt(onsets_counts)
    onsets_dens = onsets_counts / np.sum(onsets_counts)
    onsets_dens_err = onsets_counts_err / np.sum(onsets_counts)

    return (
        onsets,
        onsets_counts,
        onsets_bins,
        onsets_counts_err,
        onsets_dens,
        onsets_dens_err,
    )


# Decomposition Function
def decompositionfunc(iso_counts, convec_mlt, iso_err):
    iso_counts = np.array(iso_counts)
    convec_mlt_dens = convec_mlt / np.sum(convec_mlt)
    chi_hist = np.inf
    dist = np.zeros(24)
    fit_hist = np.zeros(24)

    n = 0
    while (np.array(iso_counts) - dist).min() > 0:
        dist = n * np.array(convec_mlt_dens)
        dist = np.round(dist, 0)
        chi_sq = chi_squared_test(iso_counts, dist, iso_err)
        if chi_sq < chi_hist:
            chi_hist = chi_sq
            fit_hist = dist
            wght = n
        n += 1

    iso_minus_convec = np.array(iso_counts) - fit_hist

    substorm_fit = iso_minus_convec
    convec_fit = convec_mlt

    # mask = np.zeros(np.shape(iso_minus_convec))
    # mask[7:17] = 1
    # # substorm_fit = np.where(mask, iso_minus_convec, 0)

    # mask = np.zeros(np.shape(iso_minus_convec))
    # mask[:7] = 1
    # mask[17:] = 1
    # convec_fit = np.where(mask, iso_minus_convec, 0) + convec_mlt

    substorm_fit_dens = substorm_fit / np.sum(substorm_fit)
    convec_fit_dens = convec_fit / np.sum(convec_fit)

    n_substorm_iso = np.sum(iso_counts) - wght
    n_convec_iso = wght

    return (
        convec_fit,
        convec_fit_dens,
        substorm_fit,
        substorm_fit_dens,
        n_substorm_iso,
        n_convec_iso,
    )

# Fitting function from DP1 and DP2 distributions
def fit_substorm_convec(observed_dist, dist_uncertainty, len):
    observed_dist = np.array(observed_dist)
    dist_uncertainty = np.array(dist_uncertainty)
    chi_hist = np.inf
    fit_hist = []

    for i in range(len):
        observed_dist_loop = observed_dist.copy()
        dist_uncertainty_loop = dist_uncertainty.copy()
        dist = np.array(substorm_fit_dens) * i + np.array(convec_fit_dens) * (len - i)
        dist_check = dist.copy()

        if np.where(observed_dist == 0)[0].size > 0:
            observed_dist_loop = observed_dist_loop[
                np.setdiff1d(np.arange(24), np.where(observed_dist == 0)[0])
            ]
            dist_check = dist_check[
                np.setdiff1d(np.arange(24), np.where(observed_dist == 0)[0])
            ]
            dist_uncertainty_loop = dist_uncertainty[
                np.setdiff1d(np.arange(24), np.where(observed_dist == 0)[0])
            ]

        chi_sq = chi_squared_test(observed_dist_loop, dist_check, dist_uncertainty_loop)
        if chi_sq < chi_hist:
            chi_hist = chi_sq
            n_substorm = i
            n_convec = len - i
            fit_hist = dist

    return fit_hist, n_substorm, n_convec


# Labelling of the MLT bins for plotting
bins = list(range(12, 24)) + list(range(0, 12))
bins = list(map(str, bins))

# %%
# SOPHIE SMU Threshold 1
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_1)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_1)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_1)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_1)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_1)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_1)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Isolated Fitting

fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_1 = onsets_mlt.size
n_substorms_smu_1 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_1 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_1 = convec_onsets_mlt.size
iso_mlt_counts_smu_1 = iso_mlt_counts
iso_mlt_dens_smu_1 = iso_mlt_dens

# %%
# SOPHIE SMU Threshold 2
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_2)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_2)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_2)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_2)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_2)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_2)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)

# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)


# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_2 = onsets_mlt.size
n_substorms_smu_2 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_2 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_2 = convec_onsets_mlt.size
iso_mlt_counts_smu_2 = iso_mlt_counts
iso_mlt_dens_smu_2 = iso_mlt_dens

# %%
# SOPHIE SMU Threshold 3
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_3)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_3)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_3)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_3)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_3)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_3)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)
# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)


# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_3 = onsets_mlt.size
n_substorms_smu_3 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_3 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_3 = convec_onsets_mlt.size
iso_mlt_counts_smu_3 = iso_mlt_counts
iso_mlt_dens_smu_3 = iso_mlt_dens

# %% SOPHIE SMU Threshold 4
# MLT Distributions

# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_4)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_4)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_4)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_4)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_4)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_4)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)
n_onsets_smu_4 = onsets_mlt.size
n_substorms_smu_4 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_4 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_4 = convec_onsets_mlt.size
iso_mlt_counts_smu_4 = iso_mlt_counts
iso_mlt_dens_smu_4 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 5
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_5)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_5)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_5)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_5)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_5)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_5)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_5 = onsets_mlt.size
n_substorms_smu_5 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_5 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_5 = convec_onsets_mlt.size
iso_mlt_counts_smu_5 = iso_mlt_counts
iso_mlt_dens_smu_5 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 6
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_6)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_6)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_6)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_6)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_6)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_6)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_6 = onsets_mlt.size
n_substorms_smu_6 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_6 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_6 = convec_onsets_mlt.size
iso_mlt_counts_smu_6 = iso_mlt_counts
iso_mlt_dens_smu_6 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 7
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_7)

# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_7)

# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_7)

# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_7)

# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_7)

# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_7)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)
# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(
    iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size
)
# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(
    comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size
)
# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = (
    fit_substorm_convec(
        after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size
    )
)
# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(
    geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size
)
n_onsets_smu_7 = onsets_mlt.size
n_substorms_smu_7 = (
    n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg
)
n_substorms_orig_smu_7 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_7 = convec_onsets_mlt.size
iso_mlt_counts_smu_7 = iso_mlt_counts
iso_mlt_dens_smu_7 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 8
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_8)
# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_8)
# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_8)
# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_8)
# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_8)
# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_8)

# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
fig.tight_layout(pad=1)

# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_8 = onsets_mlt.size
n_substorms_smu_8 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_8 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_8 = convec_onsets_mlt.size
iso_mlt_counts_smu_8 = iso_mlt_counts
iso_mlt_dens_smu_8 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 9
# MLT Distributions
# All onsets
(
    onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_9)
# Isolated
(
    iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_9)
# Compound
(
    comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_9)
# Convection expansions
(
    convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_9)
# After convection expansions
(
    after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_9)
# GEG expansions
(
    geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_9)
# Fitting
# Decomposition
(
    convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)
# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_9 = onsets_mlt.size
n_substorms_smu_9 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_9 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_9 = convec_onsets_mlt.size
iso_mlt_counts_smu_9 = iso_mlt_counts
iso_mlt_dens_smu_9 = iso_mlt_dens

# %% SOPHIE SMU Threshhold 10
# MLT Distributions
# All onsets
(onsets_mlt,
    onsets_mlt_counts,
    onsets_mlt_bins,
    onsets_mlt_counts_err,
    onsets_mlt_dens,
    onsets_mlt_dens_err,
) = mlt_distribution(expansiondf_smu_10)

# Isolated
(iso_onsets_mlt,
    iso_mlt_counts,
    iso_mlt_bins,
    iso_mlt_counts_err,
    iso_mlt_dens,
    iso_mlt_dens_err,
) = mlt_distribution(iso_onsets_smu_10)

# Compound
(comp_onsets_mlt,
    comp_mlt_counts,
    comp_mlt_bins,
    comp_mlt_counts_err,
    comp_mlt_dens,
    comp_mlt_dens_err,
) = mlt_distribution(comp_onsets_smu_10)

# Convection expansions
(convec_onsets_mlt,
    convec_mlt_counts,
    convec_mlt_bins,
    convec_mlt_counts_err,
    convec_mlt_dens,
    convec_mlt_dens_err,
) = mlt_distribution(convec_expansiondf_smu_10)

# After convection expansions
(after_convec_mlt,
    after_convec_mlt_counts,
    after_convec_mlt_bins,
    after_convec_mlt_counts_err,
    after_convec_mlt_dens,
    after_convec_mlt_dens_err,
) = mlt_distribution(onsets_after_convec_smu_10)

# GEG expansions
(geg_mlt,
    geg_mlt_counts,
    geg_mlt_bins,
    geg_mlt_counts_err,
    geg_mlt_dens,
    geg_mlt_dens_err,
) = mlt_distribution(gegdf_smu_10)

# Fitting
# Decomposition
(convec_fit,
    convec_fit_dens,
    substorm_fit,
    substorm_fit_dens,
    n_substorm_iso,
    n_convec_iso,
) = decompositionfunc(iso_mlt_counts, convec_mlt_counts, iso_mlt_counts_err)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, substorm_fit_dens, color=colormap[7], label="DP1 Fit")
ax.plot(np.arange(24) + 0.5, convec_fit_dens, color=colormap[8], label="DP2 Fit")
ax.set_xlabel("MLT")
ax.set_ylabel("Density")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=1)
# Isolated Fitting
fit_hist_iso, n_substorm_iso, n_convec_iso = fit_substorm_convec(iso_mlt_counts, iso_mlt_counts_err, iso_onsets_mlt.size)

# Compound Fitting
fit_hist_comp, n_substorm_comp, n_convec_comp = fit_substorm_convec(comp_mlt_counts, comp_mlt_counts_err, comp_onsets_mlt.size)

# After Convection Expansion Fitting
fit_hist_after_convec, n_substorm_after_convec, n_convec_after_convec = fit_substorm_convec(after_convec_mlt_counts, after_convec_mlt_counts_err, after_convec_mlt.size)

# GEG Expansion Fitting
fit_hist_geg, n_substorm_geg, n_convec_geg = fit_substorm_convec(geg_mlt_counts, geg_mlt_counts_err, geg_mlt.size)

n_onsets_smu_10 = onsets_mlt.size
n_substorms_smu_10 = (n_substorm_iso + n_substorm_comp + n_substorm_after_convec + n_substorm_geg)
n_substorms_orig_smu_10 = onsets_mlt.size - convec_onsets_mlt.size
n_convec_smu_10 = convec_onsets_mlt.size
iso_mlt_counts_smu_10 = iso_mlt_counts
iso_mlt_dens_smu_10 = iso_mlt_dens

# %%
# Number of Substorms vs Threshold
n_events_all_thresh = np.array(
    [
        n_onsets_smu_1,
        n_onsets_smu_2,
        n_onsets_smu_3,
        n_onsets_smu_4,
        n_onsets_smu_5,
        n_onsets_smu_6,
        n_onsets_smu_7,
        n_onsets_smu_8,
        n_onsets_smu_9,
        n_onsets_smu_10,
    ]
)
n_substorms_all_thresh = np.array(
    [
        n_substorms_smu_1,
        n_substorms_smu_2,
        n_substorms_smu_3,
        n_substorms_smu_4,
        n_substorms_smu_5,
        n_substorms_smu_6,
        n_substorms_smu_7,
        n_substorms_smu_8,
        n_substorms_smu_9,
        n_substorms_smu_10,
    ]
)
n_substorms_all_thresh_orig = np.array(
    [
        n_substorms_orig_smu_1,
        n_substorms_orig_smu_2,
        n_substorms_orig_smu_3,
        n_substorms_orig_smu_4,
        n_substorms_orig_smu_5,
        n_substorms_orig_smu_6,
        n_substorms_orig_smu_7,
        n_substorms_orig_smu_8,
        n_substorms_orig_smu_9,
        n_substorms_orig_smu_10,
    ]
)
n_convec_all_thresh = np.array(
    [
        len(convec_expansiondf_smu_1),
        len(convec_expansiondf_smu_2),
        len(convec_expansiondf_smu_3),
        len(convec_expansiondf_smu_4),
        len(convec_expansiondf_smu_5),
        len(convec_expansiondf_smu_6),
        len(convec_expansiondf_smu_7),
        len(convec_expansiondf_smu_8),
        len(convec_expansiondf_smu_9),
        len(convec_expansiondf_smu_10),
    ]
)

n_years = 23

# Plotting Number of SML substorm estimation vs threshold per year
fig, ax = plt.subplots(dpi=300)

ax.plot(
    np.arange(1, 11), 
    n_events_all_thresh/n_years, 
    color=colormap[0], 
    label="Candidate Expansion Onsets"
)
ax.plot(
    np.arange(1, 11),
    n_substorms_all_thresh_orig/n_years,
    color=colormap[1],
    label="SOPHIE identified substorms",
)
ax.plot(
    np.arange(1, 11),
    n_convec_all_thresh/n_years,
    color=colormap[3],
    label="SOPHIE identified convection enhancements",
)
ax.plot(
    np.arange(1, 11),
    n_substorms_all_thresh/n_years,
    color=colormap[7],
    label="DP1 Perturbations",
    ls="--",
)
ax.plot(
    np.arange(1, 11),
    (n_events_all_thresh - n_substorms_all_thresh)/n_years,
    color=colormap[8],
    label="DP2 Perturbations",
    ls="--",
)
ax.set_xticks(np.arange(1, 11))
ax.set_xlabel("SML/SMU Ratio Threshold")
ax.set_ylabel("Number of Events per Year")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
ax.set_ylim(0, 1500)
fig.tight_layout(pad=1)


# Plotting SML substorm estimation vs threshold
iso_mlt_counts_all_thresh = [
    iso_mlt_counts_smu_1,
    iso_mlt_counts_smu_2,
    iso_mlt_counts_smu_3,
    iso_mlt_counts_smu_4,
    iso_mlt_counts_smu_5,
    iso_mlt_counts_smu_6,
    iso_mlt_counts_smu_7,
    iso_mlt_counts_smu_8,
    iso_mlt_counts_smu_9,
    iso_mlt_counts_smu_10,
]

iso_mlt_density_all_thresh = [
    iso_mlt_dens_smu_1,
    iso_mlt_dens_smu_2,
    iso_mlt_dens_smu_3,
    iso_mlt_dens_smu_4,
    iso_mlt_dens_smu_5,
    iso_mlt_dens_smu_6,
    iso_mlt_dens_smu_7,
    iso_mlt_dens_smu_8,
    iso_mlt_dens_smu_9,
    iso_mlt_dens_smu_10,
]


fig, ax = plt.subplots(dpi=300)

for i, val in enumerate(iso_mlt_counts_all_thresh):
    ax.plot(np.arange(24) + 0.5, val, label=f"Threshold {i+1}")
ax.set_xlabel("MLT")
ax.set_ylabel("Counts")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
fig.tight_layout(pad=1)

fig, ax = plt.subplots(dpi=300)
for i, val in enumerate(iso_mlt_density_all_thresh):
    ax.plot(np.arange(24) + 0.5, val, label=f"Threshold {i+1}")
ax.set_xlabel("MLT")
ax.set_ylabel("Probability")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
fig.tight_layout(pad=1)


# Ratio differences
iso_mlt_diff_ratio = (np.array(iso_mlt_counts_smu_10)/np.array(iso_mlt_counts_smu_2))

fig, ax = plt.subplots(dpi=300)
ax.plot(np.arange(24) + 0.5, iso_mlt_diff_ratio)
ax.set_xlabel("MLT")
ax.set_ylabel("N_events Threshold 10/\nN_events Threshold 2")
ax.set_xticks(range(24))
ax.set_xticklabels(bins)

fig.tight_layout(pad=1)

# %%
