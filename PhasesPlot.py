# %% # Importing Modules
# Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns

sns.set_theme(
    context="notebook",
    style="ticks",
    palette="colorblind",
)
colormap = sns.color_palette("colorblind")

cm=1/2.54

# %% Loading in Substorm Data
# Loading in Substorm Data

# Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT90_1996-2024_NewFlag.csv")
sophiedf["Date_UTC"] = pd.to_datetime(sophiedf["Date_UTC"])
sophiedf["Duration"] = np.append(np.diff(sophiedf["Date_UTC"].to_numpy()), 0)
sophiedf = sophiedf[sophiedf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)
if "Delbay" in sophiedf.columns:
    sophiedf.rename(columns={"Delbay": "DeltaSML"}, inplace=True)
if "SML Val at End" in sophiedf.columns:
    sophiedf.rename(columns={"SML Val at End": "SMLatEnd"}, inplace=True)

sophiedf["DeltaSML"] = pd.to_numeric(sophiedf["DeltaSML"], errors="coerce")
sophiedf = sophiedf.loc[2:].reset_index(drop=True)

sophiedf['Flag'] = sophiedf['Flag'].apply(lambda x: 1 if x > 0 else 0)

# Loading in SME Data
smedf = pd.read_csv("Data/SMEdata.txt")
smedf["Date_UTC"] = pd.to_datetime(smedf["Date_UTC"])
smedf = smedf[smedf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)

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

# Finding last onset of compound chain that are ended by a convection interval
compend_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
for i in range(len(sophiedf["Date_UTC"]) - 2):
    if (
        (sophiedf.iloc[i]["Phase"] == 2)
        and (sophiedf.iloc[i]["NewFlag"] == 0)
        and (sophiedf.iloc[i + 2]["NewFlag"] == 1)
    ):
        compend_arr[i] = 1
        continue
    else:
        compend_arr[i] = 0
        continue
sophiedf["OnsetBeforeConvection"] = compend_arr

# %%
np.intersect1d(np.where(sophiedf["Isolated"] == 1), np.where(sophiedf["NewFlag"] == 1))[0:20]
# %%
# Period of interest 
start = "1997-01-10 14:00"
end = "1997-01-11 02:00"

phasesindices = sophiedf[sophiedf["Date_UTC"].between(start, end, inclusive="both")].index.to_numpy()
phasesindices = np.concatenate(([phasesindices[0]-1],phasesindices,[phasesindices[-1]+1]))

sme_slice = smedf[smedf["Date_UTC"].between(start, end, inclusive="both")].copy()
sophie_slice = sophiedf.iloc[phasesindices].copy()

# Plotting
fig, ax = plt.subplots(dpi=300, sharex=True, figsize=(25*cm, 10*cm))

ax.plot(sme_slice["Date_UTC"], sme_slice["SML"], label="SML", color='k', linestyle='-')
ax.plot(sme_slice["Date_UTC"], sme_slice["SMU"], label="SMU", color='k', linestyle='--')
ax.plot(sme_slice["Date_UTC"], np.zeros_like(sme_slice["SML"]), color='k', linestyle=':')

for index, row in sophie_slice.iloc[:-1].iterrows():
    if row["Phase"] == 1:
        ax.axvspan(row["Date_UTC"], sophie_slice.loc[index+1]["Date_UTC"], facecolor=colormap[2], alpha=0.5,label="Growth")
    if row["Phase"] == 2 and row["Flag"] == 0:
        if row["Isolated"] == 1:
            ax.axvspan(row["Date_UTC"], sophie_slice.loc[index+1]["Date_UTC"], facecolor=colormap[1], alpha=0.5,label="Isolated Expansion", hatch="\\\\", edgecolor="k")
        if row["Compound"] == 1:
            ax.axvspan(row["Date_UTC"], sophie_slice.loc[index+1]["Date_UTC"], facecolor=colormap[4], alpha=0.5,label="Compound Expansion", hatch="//", edgecolor="k")
    if row["Phase"] == 3 and row["Flag"] == 0:
        ax.axvspan(row["Date_UTC"], sophie_slice.loc[index+1]["Date_UTC"], facecolor=colormap[0], alpha=0.5,label="Recovery", hatch="o", edgecolor="k")
    if row["Flag"] == 1:
        ax.axvspan(row["Date_UTC"], sophie_slice.loc[index+1]["Date_UTC"], facecolor="k", alpha=0.3,label="Convection Interval")

ax.set_xlabel("Time (UTC)")
ax.set_ylabel("SME (U\L) (nT)")
ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=15))
ax.xaxis.set_major_locator(dates.MinuteLocator(interval=120))
ax.xaxis.set_major_formatter(dates.DateFormatter("%Y/%m/%d\n%H:%M"))

handles, labels = ax.get_legend_handles_labels()

handles = [handles[i] for i in sorted(labels.index(elem) for elem in set(labels))]
labels = [labels[i] for i in sorted(labels.index(elem) for elem in set(labels))]

ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1))

# %%
