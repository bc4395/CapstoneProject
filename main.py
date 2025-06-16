import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def calculate_shear_stress(Q, R_Init, k, K, n, positions, radii):
    """
    Calculate the shear stress along a conical nozzle for a non-Newtonian fluid at given radial positions.
    
    Parameters:
    Q : float, volumetric flow rate (m^3/s)
    R_0 : float, inlet radius of the nozzle (m)
    k : float, taper factor (m/m)
    K : float, consistency index (Pa·s^n)
    n : float, power-law index
    positions : array, axial positions along the nozzle (m)
    radii : array, radial positions at each axial position (m)
    
    Returns:
    tau_profile : 2D array, shear stress at each position and radius (Pa)
    """
    # Create 2D mesh for axial and radial positions
    Z, R = np.meshgrid(positions, radii)
    # Local radius at each axial position
    R_z = R_Init - k * Z
    # Avoid division by zero or negative values
    R_z = np.maximum(R_z, 1e-10)
    # Shear rate at each (r, z) position, scaled by r/R_z
    shear_rate = (4 * Q / (np.pi * R_z**3)) * (R / R_z)
    # Shear stress using power-law model
    shear_stress = K * shear_rate**n
    return shear_stress

# Nozzle geometry
R_In = 0.005     # m
R_Out = 0.00022   # m
L = 0.0314      # m
R_avg = (R_In + R_Out) / 2
k = (R_In - R_Out) / L

# Load the CSV
df = pd.read_csv("A4C4.csv")

# Extract shear rate (SR) and viscosity (Vis)
shear_rate = df['SR'].values
viscosity = df['Vis'].values

# Transform data to logarithmic scale
log_shear_rate = np.log10(shear_rate)
log_viscosity = np.log10(viscosity)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(log_shear_rate, log_viscosity)

# Calculate power-law parameters
n = slope + 1
K = 10 ** intercept

# Define input pressure
input_pressure = int(input("Enter pressure used to print (psi): "))\

# Convert psi to Pa
pressure_pa = input_pressure * 6894.76  # psi to Pa

# Volumetric flow rate calculation
Q = (np.pi * R_avg**3 / (4 * K)) * (pressure_pa / L)**(1/n)  # Volumetric flow rate (m^3/s)

# Nozzle position and radial coordinates
positions_m = np.linspace(0, L, 100)
radii = np.linspace(0, R_In, 50)  # Radial positions from centerline to max inlet radius
R_z = R_In - k * positions_m  # Local radius at each axial position

# Calculate 2D shear stress for heatmap
shear_stress_2d = calculate_shear_stress(Q, R_In, k, K, n, positions_m, radii)

# Calculate wall shear stress for the profile plot (at r = R_z)
shear_stress_tau = calculate_shear_stress(Q, R_In, k, K, n, positions_m, R_z)

# User printing
print("\nNozzle Dimensions:")
print("Outlet Radius (R_L): {:.4f} m".format(R_Out))
print("\nResults:")
print(f"Consistency Index (K): {K:.4f} Pa·s^n")
print(f"Power-law Index (n): {n:.4f}")
print("Regression R-squared value: {:.4f}".format(r_value**2))
print(f"Pressure Input: {input_pressure} Pa")
print(f"Calculated Volumetric Flow Rate (Q): {Q:.6f} m^3/s")
print(f"Max Shear Stress: {shear_stress_tau[-1, :].max():.4f} Pa at position {positions_m[np.argmax(shear_stress_tau[-1, :])]:.4f} m")

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Shear Stress Profile (wall shear stress)
ax1.plot(positions_m, shear_stress_tau[-1, :], color='blue')  # Use last radial position (wall)
ax1.set_xlabel('Position along nozzle (m)')
ax1.set_ylabel('Shear Stress (Pa)')
ax1.set_title('Wall Shear Stress Profile')
ax1.grid(True)

# Plot 2: Heatmap of Shear Stress
heatmap = ax2.pcolormesh(positions_m, radii, shear_stress_2d, cmap='hot', shading='auto')
ax2.set_xlabel('Position along nozzle (m)')
ax2.set_ylabel('Radial position (m)')
ax2.set_title('Shear Stress Heatmap (Pa)')
fig.colorbar(heatmap, ax=ax2, label='Shear Stress (Pa)')

plt.tight_layout()
plt.show()