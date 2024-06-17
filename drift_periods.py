# %% Package Imports

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(context="talk",style="ticks",palette="colorblind")
colors = sns.color_palette("colorblind",as_cmap=True)

# %%
# Drift periods
lshells = np.linspace(2,8,50)

def drift_period(x,pitch_angle):
    pi = np.pi
    B_E = 3.11e-5
    R_E_2 = 6.378e6 ** 2
    charge = 1.6e-19
    energy = 1e3 * charge
    pitch_angle_rad = pitch_angle * pi/180
    const = (0.35 + 0.15 * np.sin(pitch_angle_rad)) ** -1 
    return (pi * charge * B_E * R_E_2 * const)/(3 * x * energy)

drift_period_90 = drift_period(lshells,90)/(60 ** 2)
drift_period_00 = drift_period(lshells,0)/(60 ** 2)

fig, ax = plt.subplots(dpi=300)

ax.plot(lshells,drift_period_00,label="0 deg pitch angle")
ax.plot(lshells,drift_period_90,label="90 deg pitch angle")
ax.set_ylabel("Drift period (hours)")
ax.set_xlabel("L shell")
ax.legend(loc='best')
ax.set_title("Drift period vs L-shell for 1 keV particle")

fig.tight_layout()

energies = np.arange(10,150)

def drift_period_energy(energy_kev,pitch_angle):
    pi = np.pi
    B_E = 3.11e-5
    R_E_2 = 6.378e6 ** 2
    charge = 1.6e-19
    energy = energy_kev*1e3 * charge
    pitch_angle_rad = pitch_angle * pi/180
    const = (0.35 + 0.15 * np.sin(pitch_angle_rad)) ** -1 
    return (pi * charge * B_E * R_E_2 * const)/(3 * 6.6 * energy)

drift_period_00 = drift_period_energy(energies, 0)/(60 ** 2)
drift_period_90 = drift_period_energy(energies, 90)/(60 ** 2)

fig, ax = plt.subplots(dpi=300)

ax.plot(energies,drift_period_00,label="0 deg pitch angle")
ax.plot(energies,drift_period_90,label="90 deg pitch angle")
ax.set_ylabel("Drift period (hours)")
ax.set_xlabel("Energy (keV)")
ax.legend(loc='best')
ax.set_title("Drift period vs Energy (keV) at L-shell of 6.6")

fig.tight_layout()

drift_periods = np.arange(10,361)

def particle_eng_for_period(d_period, pitch_angle):
    pi = np.pi
    B_E = 3.11e-5
    R_E_2 = 6.378e6 ** 2
    charge = 1.6e-19
    pitch_angle_rad = pitch_angle * pi/180
    const = (0.35 + 0.15 * np.sin(pitch_angle_rad)) ** -1 
    d_period_seconds = d_period * (60 ** 1)
    return (pi * charge * B_E * R_E_2 * const)/(3 * 6.6 * d_period_seconds) 

energy_00 = particle_eng_for_period(drift_periods,0)/(1e3*1.6e-19)
energy_90 = particle_eng_for_period(drift_periods, 90)/(1e3*1.6e-19)

fig, ax = plt.subplots(dpi=300)

ax.plot(drift_periods,energy_00, label="0 deg pitch angle")
ax.plot(drift_periods,energy_90, label="90 deg pitch angle")
ax.set_ylabel("Energy (keV)")
ax.set_xlabel("Drift period (mins)")
ax.legend(loc='best')
ax.set_title("Energy (keV) vs Drift period at L-shell of 6.6")

fig.tight_layout()

def drift_speed(energy_kev, pitchangle):
    pi = np.pi
    B_E = 3.11e-5
    R_E = 6.378e6
    charge = 1.6e-19
    energy = energy_kev * 1e3 * charge
    pitch_angle_rad = pitchangle * pi/180
    const = (0.35 + 0.15 * np.sin(pitch_angle_rad))
    return (6* (6.6 ** 2) * energy * const)/(charge * B_E * R_E)

drift_speeds_00 = drift_speed(energies,0) / ((np.pi * 6.6 * 6.38e6)/43200)
drift_speeds_90 = drift_speed(energies,90) / ((np.pi * 6.6 * 6.38e6)/43200) 

fig, ax = plt.subplots(dpi=300)

ax.plot(energies,drift_speeds_00,label="0 deg pitch angle")
ax.plot(energies,drift_speeds_90,label="90 deg pitch angle")
ax.set_ylabel("Drift speed (MLT/hour)")
ax.set_xlabel("Energy (keV)")
ax.legend(loc='best')
ax.set_title("Drift speed vs Energy (keV) at L-shell of 6.6")

# %%