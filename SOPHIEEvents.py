# %% # Importing Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns

# Housekeeping
sns.set_theme(context="paper",style="whitegrid",palette="colorblind")
colormap = sns.color_palette("colorblind")

# %% Loading in Substorm Data

# Loading in SOPHIE Data
sophiedf = pd.read_csv("Data/SOPHIE_EPT90_1996-2021.txt")
sophiedf["Date_UTC"] = pd.to_datetime(sophiedf["Date_UTC"])
sophiedf["Phase Duration"] = np.append(np.diff(sophiedf["Date_UTC"].to_numpy()), 0)
sophiedf = sophiedf[sophiedf["Date_UTC"].between("1997", "2020", inclusive="left")].reset_index(drop=True)
if "Delbay" in sophiedf.columns:
    sophiedf.rename(columns={"Delbay": "DeltaSML"}, inplace=True)
if "SML Val at End" in sophiedf.columns:
    sophiedf.rename(columns={"SML Val at End": "SMLatEnd"}, inplace=True)
sophiedf["DeltaSML"] = pd.to_numeric(sophiedf["DeltaSML"], errors="coerce")
sophiedf = sophiedf.loc[2:].reset_index(drop=True)

sophiedf["Flag"] = sophiedf["Flag"].replace(4, 0)
sophiedf["Flag"] = sophiedf["Flag"].replace([1, 2, 3, 5, 6, 7], 1)

# %% SOPHIE Phases

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
# Excluding expansion phases directly before growth phases
expansionbeforegrowth_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
for i in range(len(sophiedf["Date_UTC"]) - 1):
    if (sophiedf.iloc[i]["Phase"] == 2) and (sophiedf.iloc[i + 1]["Phase"] == 1):
        expansionbeforegrowth_arr[i] = 1

for i in reversed(range(len(sophiedf["Date_UTC"]) - 2)):
    if (sophiedf.iloc[i]["Phase"] == 2) and (expansionbeforegrowth_arr[i + 2] == 1):
        expansionbeforegrowth_arr[i] = 1

# Excluding expansion phases directly after recovery phases that follow a growth phase
expansionafterGR_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
for i in range(2, len(sophiedf["Date_UTC"])):
    if (
        (sophiedf.iloc[i]["Phase"] == 2)
        and (sophiedf.iloc[i - 1]["Phase"] == 3)
        and (sophiedf.iloc[i - 2]["Phase"] == 1)
    ):
        expansionafterGR_arr[i] = 1

for i in range(2, len(sophiedf["Date_UTC"])):
    if (sophiedf.iloc[i]["Phase"] == 2) and (expansionafterGR_arr[i - 2] == 1):
        expansionafterGR_arr[i] = 1

compound_arr = np.zeros(len(sophiedf["Date_UTC"]), dtype=int)
compound_arr[np.setdiff1d(np.where(sophiedf["Phase"] == 2), np.where(iso_arr == 1))] = 1
compound_arr[np.where(expansionbeforegrowth_arr == 1)] = 0
compound_arr[np.where(expansionafterGR_arr == 1)] = 0
sophiedf["Compound Onset"] = compound_arr

# Excluding onsets after a convection interval
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

# %% SOPHIE Event types

# Only Expansion Phases
expansiondf = sophiedf.iloc[np.where(sophiedf["Phase"] == 2)].reset_index(drop=True)

# Only Convection Expansions
convec_expansiondf = expansiondf.iloc[np.where(expansiondf["Flag"] == 1)]

# Isolated Onsets
isolated = np.intersect1d(np.where(expansiondf["Isolated Onset"] == 1), np.where(expansiondf["NewFlag"] == 0))
iso_onsets = expansiondf.iloc[isolated]

# Isolated Onsets flagged as convection
iso_onsets_convec = np.intersect1d(np.where(expansiondf["Isolated Onset"] == 1), np.where(expansiondf["Flag"] == 1))
iso_onsets_convec = expansiondf.iloc[iso_onsets_convec]

# %% SOPHIE Isolated Onset at 23-00 MLT

type1_onsets = iso_onsets.iloc[np.where(iso_onsets["MLT"] > 23)].reset_index(drop=True)
type1_onsets.set_index("Date_UTC", inplace=True)
type1_onsets = type1_onsets.between_time("02:30", "05:00")
type1_onsets.reset_index(inplace=True)
type1_onsets = type1_onsets[type1_onsets["Date_UTC"].between("2000-05-18", "2003", inclusive="left")].reset_index(drop=True)

# %% SOPHIE Isolated Onset at 04-09 MLT

type2_onsets = iso_onsets.iloc[np.intersect1d(np.where(iso_onsets["MLT"] >= 4), np.where(iso_onsets["MLT"] <= 9))].reset_index(drop=True)
type2_onsets.set_index("Date_UTC", inplace=True)
type2_onsets = type2_onsets.between_time("02:30", "05:00")
type2_onsets.reset_index(inplace=True)
type2_onsets = type2_onsets[type2_onsets["Date_UTC"].between("2000-05-18", "2003", inclusive="left")].reset_index(drop=True)

# %% SOPHIE Isolated Convection Onset at 23-00 MLT

type3_onsets = iso_onsets_convec.iloc[np.where(iso_onsets_convec["MLT"] > 23)].reset_index(drop=True)
type3_onsets.set_index("Date_UTC", inplace=True)
type3_onsets = type3_onsets.between_time("02:30", "05:00")
type3_onsets.reset_index(inplace=True)
# type3_onsets = type3_onsets[type3_onsets["Date_UTC"].between("2000-05-18", "2003", inclusive="left")].reset_index(drop=True)
type3_onsets['DeltaSML'] = pd.to_numeric(type3_onsets['DeltaSML'], errors='coerce')
type3_onsets['SMLatEnd'] = pd.to_numeric(type3_onsets['SMLatEnd'], errors='coerce')
np.sort(np.abs(type3_onsets['SMLatEnd']) - np.abs(type3_onsets['DeltaSML']))

# %% Data for Dovile

# Isolated Onsets
event1_id = np.intersect1d(np.where(sophiedf["MLT"] >= 23),np.intersect1d(np.where(sophiedf["Isolated Onset"]==1),np.where(sophiedf["Flag"]==0)))
starttimes = sophiedf.iloc[event1_id]["Date_UTC"].to_numpy()
endtimes = sophiedf.iloc[event1_id+2]['Date_UTC'].to_numpy()
df = pd.DataFrame({"Start": starttimes, "End": endtimes})
# df.to_csv("Outputs/SubstormEventTimes.csv", index=True)

# Convection Onsets
event2_id = np.intersect1d(np.where(sophiedf["Isolated Onset"]==1),np.where(sophiedf["Flag"]==1))
starttimes = sophiedf.iloc[event2_id]["Date_UTC"].to_numpy()
endtimes = sophiedf.iloc[event2_id+2]['Date_UTC'].to_numpy()
df = pd.DataFrame({"Start": starttimes, "End": endtimes})
# df.to_csv("Outputs/ConvectionEventTimes.csv", index=True)
# %%
