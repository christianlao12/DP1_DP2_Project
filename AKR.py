# %% import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="colorblind",
)
colormap = sns.color_palette("colorblind", as_cmap=True)
# %%
akr98 = pd.read_csv("Data/AKR/waters_intensity_with_location_two_freq_bands_1998.csv")
akr98 = akr98[["datetime_ut", "P_Wsr-1_100_650_kHz", "P_Wsr-1_30_100_kHz"]]
akr98.rename(columns={"datetime_ut": "Date_UTC"}, inplace=True)
akr98["Date_UTC"] = pd.to_datetime(akr98["Date_UTC"]).round('1s').apply(lambda x: x.ceil('1min'))
akr98
# %%
x = akr98["Date_UTC"]
y_upper = akr98["P_Wsr-1_100_650_kHz"].to_numpy()
# y_lower = akr98["P_Wsr-1_30_100_kHz"]
fig, ax = plt.subplots(dpi=300)
ax.plot(x, y_upper)
# ax.plot(x,-y_lower)
ax.set_xlabel("Date")
ax.set_ylabel("Mean Integrated Power 100-650kHz (W/sr)")

# %%
print(f"AKR Main Band Intensity: \nMin: {np.min(y_upper)} \nLower Quartile: {np.quantile(y_upper[np.nonzero(y_upper)],.25)} \nMedian: {np.quantile(y_upper[np.nonzero(y_upper)],.5)} \nUpper Quartile: {np.quantile(y_upper[np.nonzero(y_upper)],.75)} \nMax: {np.max(y_upper)} \n% Zero vals: {(np.size(y_upper)-np.count_nonzero(y_upper))/np.size(y_upper)*100}")
# %%
