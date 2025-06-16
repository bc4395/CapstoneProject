import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Load the CSV
df = pd.read_csv("A4C4.csv")

# Calculate shear stress (τ = SR * Vis)
df['Tau'] = df['SR'] * df['Vis']

# Take logs
log_SR = np.log(df['SR'])
log_Tau = np.log(df['Tau'])

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(log_SR, log_Tau)

# Power-law index (n) and Consistency index (K)
n = slope
K = np.exp(intercept)

def calculate_shear_stress(Q, R_0, k, L, K, n, positions):
    """
    Calculate the shear stress along a conical nozzle for a non-Newtonian fluid.
    
    Parameters:
    Q : float, volumetric flow rate (m^3/s)
    R_0 : float, inlet radius of the nozzle (m)
    k : float, taper factor (m/m)
    L : float, length of the nozzle (m)
    K : float, consistency index (Pa·s^n)
    n : float, power-law index
    positions : array, positions along the nozzle (m)
    
    Returns:
    tau_profile : array, shear stress at each position (Pa)
    """
    # Calculate wall shear stress using the provided equation
    shear_stress = K * (4 * Q / (np.pi * (R_0 - k * positions)**3))**n
    
    return shear_stress

# Parameters
Q = 1e-5  # Volumetric flow rate (m^3/s)
R_0 = 0.01  # Inlet radius (m)
R_L = 0.005  # Outlet radius (m)
L = 0.01  # Nozzle length (m)
k = (R_0 - R_L) / L  # Taper factor (m/m)

# Positions along the nozzle
positions_m = np.linspace(0, L, 100)

# Calculate shear stress
shear_stress_tau = calculate_shear_stress(Q, R_0, k, L, K, n, positions_m)

# Convert positions to mm for plotting
positions_mm = positions_m * 1000

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(positions_mm, shear_stress_tau, label='Shear Stress (Pa)', color='blue')
plt.xlabel('Position along nozzle (mm)')
plt.ylabel('Shear Stress (Pa)')
plt.title('Shear Stress in Conical Nozzle (Alginate-Carboxymethyl)')
plt.grid(True)
plt.legend()
plt.text(8, 180, f'Maximum Shear Stress: {shear_stress_tau.max():.2f} Pa', fontsize=10, color='gray')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()