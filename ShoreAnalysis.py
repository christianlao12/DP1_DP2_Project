# %% Imports
import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker, colors
from cmcrameri import cm

sns.set_theme(context="paper",style="whitegrid",palette="colorblind",)
colormap = sns.color_palette("colorblind")
# %% Load data
ds01file = nc.Dataset('Data/ShoreData/Shore-ds01.nc')
ds02file = nc.Dataset('Data/ShoreData/Shore-ds02.nc')
ds04file = nc.Dataset('Data/ShoreData/Shore-ds04.nc')
ds06file = nc.Dataset('Data/ShoreData/Shore-ds06.nc')

# Locations
colats = ds01file.variables['bin_centroids_colatitude'][:].data[0]
lons = ds01file.variables['bin_centroids_longitude'][:].data[0]
lats = 90 - colats

unique_colats = np.unique(colats)
unique_lats = np.unique(lats)

# %% Polar plot of Feb 2001
dp2_200102 = ds04file.variables['eig_s_200102_mode01'][:].data[0]
dp1_200102 = ds04file.variables['eig_s_200102_mode02'][:].data[0]

dp2_200102_max = dp2_200102[np.nanargmax(np.abs(dp2_200102))]
dp2_200102_norm = dp2_200102/dp2_200102_max
dp2_200102_norm_theta = dp2_200102_norm[559:559*2]

dp1_200102_max = dp1_200102[np.nanargmax(np.abs(dp1_200102))]
dp1_200102_norm = dp1_200102/dp1_200102_max
dp1_200102_norm_theta = dp1_200102_norm[559:559*2]

# Location of the bins
theta = np.deg2rad(lons)
r = colats

# Location of lines
theta_lines = np.deg2rad(np.linspace(0, 360, 120))
r_lines_1 = np.ones_like(theta_lines) * (90 - 64)
r_lines_2 = np.ones_like(theta_lines) * (90 - 73)

# Colorbar bounds
bounds = np.linspace(-1, 1, 31)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

mltlabels = ['00', '03', '06', '09', '12', '15', '18', '21']

# DP2 Feb 2001 Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300,layout='constrained')
cmap = ax.scatter(theta, r, c=dp2_200102_norm_theta,norm=norm,cmap=cm.vik)
ax.plot(theta_lines, r_lines_1, color=colormap[2],ls='-',linewidth=2)
ax.plot(theta_lines, r_lines_2, color=colormap[2],ls='-',linewidth=2)
ax.set_theta_zero_location('S')
fig.colorbar(cmap,ax=ax, label='Spatial Amplitude Factor',)
ax.set_title('DP2 Feb 2001')
ax.set_xticklabels(mltlabels)
ax.set_yticklabels([])

# DP1 Feb 2001 Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300,layout='constrained')
cmap = ax.scatter(theta, r, c=dp1_200102_norm_theta, norm=norm,cmap=cm.vik)
ax.plot(theta_lines, r_lines_1, color=colormap[7],ls='-.',linewidth=2)
ax.plot(theta_lines, r_lines_2, color=colormap[7],ls='-.',linewidth=2)
ax.set_theta_zero_location('S')
fig.colorbar(cmap,ax=ax, label='Spatial Amplitude Factor')
ax.set_title('DP1 Feb 2001')
ax.set_xticklabels(mltlabels)
ax.set_yticklabels([])

# %% Load Full  DP2 and DP1

# Load DP2 Spatial Patterns
dp2_mode_years = ds06file.variables['DP2_group_years'][:].data[0]
dp2_mode_months = ds06file.variables['DP2_group_months'][:].data[0]
dp2_mode_nums = ds06file.variables['DP2_group_modes'][:].data[0]  
dp2_mode_string = [f'eig_s_{dp2_mode_years[i]:g}{dp2_mode_months[i]:02g}_mode{dp2_mode_nums[i]:02g}' for i in range(len(dp2_mode_years))]

dp2_amps = np.empty((len(dp2_mode_string),1677))
for i, mode in enumerate(dp2_mode_string):
    data = ds04file.variables[f'{mode}'][:].data[0]
    datamax = data[np.nanargmax(np.abs(data))]
    data = data/datamax
    dp2_amps[i] = data

dp2_amps_mean = np.nanmean(dp2_amps, axis=0)

# Load DP1 Spatial Patterns
dp1_mode_years = ds06file.variables['DP1_group_years'][:].data[0]
dp1_mode_months = ds06file.variables['DP1_group_months'][:].data[0]
dp1_mode_nums = ds06file.variables['DP1_group_modes'][:].data[0] 
dp1_mode_string = [f'eig_s_{dp1_mode_years[i]:g}{dp1_mode_months[i]:02g}_mode{dp1_mode_nums[i]:02g}' for i in range(len(dp1_mode_years))]

dp1_amps = np.empty((len(dp1_mode_string),1677))
for i, mode in enumerate(dp1_mode_string):
    data = ds04file.variables[f'{mode}'][:].data[0]
    datamax = data[np.nanargmax(np.abs(data))]
    data = data/datamax
    dp1_amps[i] = data

dp1_amps_mean = np.nanmean(dp1_amps, axis=0)

# %% Polar plots of DP1 and DP2
dp2_amps_mean_max = dp2_amps_mean[np.nanargmax(np.abs(dp2_amps_mean))]
dp2_amps_mean_norm = dp2_amps_mean/dp2_amps_mean_max
dp2_amps_mean_theta = dp2_amps_mean_norm[559:559*2]

dp1_amps_mean_max = dp1_amps_mean[np.nanargmax(np.abs(dp1_amps_mean))]
dp1_amps_mean_norm = dp1_amps_mean/dp1_amps_mean_max
dp1_amps_mean_theta = dp1_amps_mean_norm[559:559*2]

# Location of the bins
theta = np.deg2rad(lons)
r = colats

# Location of lines
theta_lines = np.deg2rad(np.linspace(0, 360, 120))
r_lines_1 = np.ones_like(theta_lines) * (90 - 64)
r_lines_2 = np.ones_like(theta_lines) * (90 - 76)

# Colorbar bounds
bounds = np.linspace(-1, 1, 21)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

# DP2 Mean Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300,layout='constrained')
cmap = ax.scatter(theta, r, c=dp2_amps_mean_theta,norm=norm,cmap=cm.vik,marker='8',s=50)
ax.plot(theta_lines, r_lines_1, color=colormap[2],ls='-',linewidth=4)
ax.plot(theta_lines, r_lines_2, color=colormap[2],ls='-',linewidth=4)
ax.set_theta_zero_location('S')
fig.colorbar(cmap,ax=ax, label='Normalised Southward Magnetic Field Perturbation',ticks=ticker.MultipleLocator(0.1))
ax.set_title('DP2')
ax.set_xticklabels(mltlabels)
ax.set_yticklabels([])

# DP1 Mean Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300,layout='constrained')
cmap = ax.scatter(theta, r, c=dp1_amps_mean_theta,norm=norm,cmap=cm.vik,marker='8',s=50)
ax.plot(theta_lines, r_lines_1, color=colormap[7],ls='-.',linewidth=4)
ax.plot(theta_lines, r_lines_2, color=colormap[7],ls='-.',linewidth=4)
ax.set_theta_zero_location('S')
fig.colorbar(cmap,ax=ax, label='Normalised Southward Magnetic Field Perturbation',ticks=ticker.MultipleLocator(0.1))
ax.set_title('DP1')
ax.set_xticklabels(mltlabels)
ax.set_yticklabels([])

# %%

# %% Latitude profiles
lat_id = 5
colat_at_colat = colats[np.where(lats==unique_lats[lat_id])]
lon_at_lat = lons[np.where(lats==unique_lats[lat_id])]
dp2_amps_at_lat = dp2_amps_mean_theta[np.where(lats==unique_lats[lat_id])]
dp1_amps_at_lat = dp1_amps_mean_theta[np.where(lats==unique_lats[lat_id])]
print("Latitude of interest:", unique_lats[lat_id])

lon_at_lat_r = np.roll(lon_at_lat, -lon_at_lat.argmin())
dp2_amps_at_lat_r = np.roll(dp2_amps_at_lat, -lon_at_lat.argmin())
dp1_amps_at_lat_r = np.roll(dp1_amps_at_lat, -lon_at_lat.argmin())

dp2_amps_3x = np.tile(dp2_amps_at_lat_r, 3)
dp1_amps_3x = np.tile(dp1_amps_at_lat_r, 3)
lon_3x = np.concatenate((lon_at_lat_r-360, lon_at_lat_r, lon_at_lat_r+360))

nbins = 24
step = 360/nbins
lon_new = np.arange(step/2, 360+step/2, step)

# Linear interpolation
dp2_amps_at_lat_new = np.interp(lon_new, lon_3x, dp2_amps_3x)
dp1_amps_at_lat_new = np.interp(lon_new, lon_3x, dp1_amps_3x)

fig, ax = plt.subplots(dpi=300)

# ax.plot(lon_at_lat_r, dp2_amps_at_lat_r, 'o', label='DP2 Profile')
# ax.plot(lon_at_lat_r, dp1_amps_at_lat_r, 'o', label='DP1 Profile')
ax.plot(lon_3x, dp2_amps_3x, label='DP2',)
ax.plot(lon_3x, dp1_amps_3x, label='DP1',)
ax.plot(lon_new, dp2_amps_at_lat_new, 'o', label='DP2 Interpolated to 24 points')
ax.plot(lon_new, dp1_amps_at_lat_new, 'o', label='DP1 Interpolated to 24 points')
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.set_xlabel('Longitude')
ax.set_ylabel('Amplitude')
ax.set_xlim(0,360)
ax.legend()

# %%
latsofinterest = unique_lats[5:9]
dp2_amps_lats = np.zeros((len(latsofinterest), lon_new.size))
dp1_amps_lats = np.zeros((len(latsofinterest), lon_new.size))
print("Latitudes of interest:", latsofinterest)

for i, lat in enumerate(latsofinterest):
    lon_at_lat = lons[np.where(lats==lat)]
    dp2_amps_at_lat = dp2_amps_mean_theta[np.where(lats==lat)]
    dp1_amps_at_lat = dp1_amps_mean_theta[np.where(lats==lat)]

    lon_at_lat_r = np.roll(lon_at_lat, -lon_at_lat.argmin())
    dp2_amps_at_lat_r = np.roll(dp2_amps_at_lat, -lon_at_lat.argmin())
    dp1_amps_at_lat_r = np.roll(dp1_amps_at_lat, -lon_at_lat.argmin())

    dp2_amps_3x = np.tile(dp2_amps_at_lat_r, 3)
    dp1_amps_3x = np.tile(dp1_amps_at_lat_r, 3)
    lon_3x = np.concatenate((lon_at_lat_r-360, lon_at_lat_r, lon_at_lat_r+360))
    
    dp2_amps_at_lat_new = np.interp(lon_new, lon_3x, dp2_amps_3x)
    dp1_amps_at_lat_new = np.interp(lon_new, lon_3x, dp1_amps_3x)

    dp2_amps_lats[i] = dp2_amps_at_lat_new
    dp1_amps_lats[i] = dp1_amps_at_lat_new

fig, ax = plt.subplots(2,1,dpi=300)

for i, lat in enumerate(latsofinterest):
    ax[0].plot(lon_new, dp2_amps_lats[i], label=f'DP2 Profile at {lat:.01f} Latitude')
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(30))
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Amplitude')
ax[0].legend(loc='center left',bbox_to_anchor=(1, 0.5))

for i, lat in enumerate(latsofinterest):
    ax[1].plot(lon_new, dp1_amps_lats[i], label=f'DP1 Profile at {lat:.01f} Latitude')
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(30))
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Amplitude')
ax[1].legend(loc='center left',bbox_to_anchor=(1, 0.5))

fig.tight_layout()

lon_new_rotated = np.roll(lon_new,lon_new.size//2)
dp2_amps_rotated = np.roll(dp2_amps_lats, lon_new.size//2, axis=1)
dp1_amps_rotated = np.roll(dp1_amps_lats, lon_new.size//2, axis=1)

bins = np.roll(np.arange(0,24)+0.5,12)
binlabels = [f'{int(i):02d}' for i in bins]

fig, ax = plt.subplots(dpi=300, figsize=(8,4))

for i, lat in enumerate(latsofinterest):
    ax.plot(np.arange(0,lon_new.size)+0.5, dp2_amps_rotated[i], label=f'DP2 Profile at {lat:.01f} Latitude',color=colormap[i])
ax.set_xlabel('MLT')
ax.set_ylabel('Shore Spatial Amplitude')
ax.set_xticks(np.arange(0,lon_new.size))
ax.set_xticklabels(binlabels)
ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))

fig, ax = plt.subplots(dpi=300, figsize=(8,4))

for i, lat in enumerate(latsofinterest):
    ax.plot(np.arange(0,lon_new.size)+0.5, dp1_amps_rotated[i], label=f'DP1 Profile at {lat:.01f} Latitude',color=colormap[i])
ax.set_xlabel('MLT')
ax.set_ylabel('Shore Spatial Amplitude')
ax.set_xticks(np.arange(0,lon_new.size))
ax.set_xticklabels(binlabels)
ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))

# dp1_df = pd.DataFrame({"MLT": bins, f"{latsofinterest[0]:.0f}": dp1_amps_rotated[0], f"{latsofinterest[1]:.0f}": dp1_amps_rotated[1], f"{latsofinterest[2]:.0f}": dp1_amps_rotated[2], f"{latsofinterest[3]:.0f}": dp1_amps_rotated[3]})
# dp2_df = pd.DataFrame({"MLT": bins, f"{latsofinterest[0]:.0f}": dp2_amps_rotated[0], f"{latsofinterest[1]:.0f}": dp2_amps_rotated[1], f"{latsofinterest[2]:.0f}": dp2_amps_rotated[2], f"{latsofinterest[3]:.0f}": dp2_amps_rotated[3]})

# dp1_df.to_csv('Data/ShoreData/ShoreSpatialAmpDP1.csv', index=False)
# dp2_df.to_csv('Data/ShoreData/ShoreSpatialAmpDP2.csv', index=False)

# %% Mean profile

dp2_ave = np.mean(dp2_amps_lats, axis=0)
dp1_ave = np.mean(dp1_amps_lats, axis=0)

fig, ax = plt.subplots(dpi=300)

ax.plot(lon_new, dp2_ave, label='DP2')
ax.plot(lon_new, dp1_ave, label='DP1')
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.set_xlabel('Longitude')
ax.set_ylabel('Amplitude')
ax.legend()

lon_new_rotated = np.roll(lon_new,lon_new.size//2)
dp2_rotated = np.roll(dp2_ave, lon_new.size//2)
dp1_rotated = np.roll(dp1_ave, lon_new.size//2)

fig, ax = plt.subplots(dpi=300, figsize=(8,4))
# Labelling of the MLT bins

ax.plot(np.arange(0,lon_new.size)+0.5, dp2_rotated, label='DP2',color=colormap[2])
ax.plot(np.arange(0,lon_new.size)+0.5, dp1_rotated, label='DP1',color=colormap[7],linestyle='-.')
ax.set_xlabel('MLT')
ax.set_ylabel('Shore Spatial Amplitude')
ax.set_xticks(np.arange(0,lon_new.size))
ax.set_xticklabels(binlabels)
ax.legend()

df = pd.DataFrame({'MLT': bins, 'DP2': dp2_rotated, 'DP1': dp1_rotated})
# df.to_csv('Data/ShoreData/ShoreSpatialAmpAve.csv', index=False)

# %%
