import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# --- Model function for curve fitting ---
def model(sr, K, n):
    """Power-law viscosity model: eta = K * shear_rate^(n-1)"""
    return K * np.power(sr, n - 1)

# --- Volumetric Flow Rate Calculation for Tapered Nozzle (Power-Law Fluid) ---
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    if r1 <= 0 or r2 <= 0 or L <= 0 or K <= 0 or n <= 0 or r1 == r2:
        raise ValueError("Invalid input: radii, length, K, and n must be positive, and r1 != r2")
    k = (r1 - r2) / L
    numerator = np.pi**n * n**n * delta_P * k * (r1 * r2)**((3*n + 1)/n)
    denominator = 2 * K * (r1**(3 + 1/n) - r2**(3 + 1/n))
    Q = (numerator / denominator)**(1/n)
    return Q

# --- Shear Stress Calculation Across Cross-Sections ---
def calculate_shear_stress_cross_section(Q, R_Init, R_Out, L, K, n, num_z=50, num_xy=100):
    if any(param <= 0 for param in [Q, R_Init, R_Out, L, K, n]):
        raise ValueError("All inputs must be positive.")
    
    Z = np.linspace(0, L, num_z)
    k = (R_Init - R_Out) / L
    R_z = np.maximum(R_Init - k * Z, 1e-10)
    
    X = np.zeros((num_xy, num_xy, num_z))
    Y = np.zeros((num_xy, num_xy, num_z))
    shear_stress = np.zeros((num_xy, num_xy, num_z))
    
    for i, z in enumerate(Z):
        x = np.linspace(-R_z[i], R_z[i], num_xy)
        y = np.linspace(-R_z[i], R_z[i], num_xy)
        X[:, :, i], Y[:, :, i] = np.meshgrid(x, y)
        R = np.sqrt(X[:, :, i]**2 + Y[:, :, i]**2)
        mask = R <= R_z[i]
        shear_rate = np.zeros_like(R)
        shear_rate[mask] = (4 * Q / (np.pi * R_z[i]**3)) * (R[mask] / R_z[i])
        shear_stress[:, :, i][mask] = K * np.abs(shear_rate[mask])**n
    
    return Z, X, Y, shear_stress

# --- Plot Cross-Sectional Shear Stress with Fixed Color Scale ---
def plot_shear_stress_cross_sections(Z, X, Y, shear_stress, num_plots=5):
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]
    
    # Global min/max for color scaling
    vmin = np.min(shear_stress)
    vmax = np.max(shear_stress)
    
    z_indices = np.linspace(0, len(Z) - 1, num_plots, dtype=int)
    
    for idx, ax in zip(z_indices, axes):
        contour = ax.contourf(
            X[:, :, idx] * 1e6, Y[:, :, idx] * 1e6, shear_stress[:, :, idx],
            levels=50, cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Z = {Z[idx] * 1e3:.2f} mm')
        ax.set_aspect('equal')

    # Shared colorbar
    cbar = fig.colorbar(contour, ax=axes, label='Shear Stress (Pa)', orientation='vertical', shrink=0.8)
    plt.suptitle('Shear Stress Distribution in Nozzle Cross-Sections', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Load and Fit A4C4 Data
    try:
        data = pd.read_csv('A4C4.csv')
        sr_data = data['SR'].values
        vis_data = data['Vis'].values
    except FileNotFoundError:
        raise FileNotFoundError("A4C4.csv not found. Ensure the file exists.")
    
    # Fit power-law model
    initial_guess = [1.0, 1.0]
    try:
        popt, _ = curve_fit(model, sr_data, vis_data, p0=initial_guess)
        K, n = popt
    except RuntimeError:
        raise RuntimeError("Curve fitting failed. Check input data or initial guess.")
    
    # Geometry Parameters
    R_In = 0.0015     # Inlet radius: 500 μm
    R_Out = 0.0004318    # Outlet radius: 220 μm
    L = 0.0314        # Nozzle length: 31.5 mm

    # Pressure Input
    try:
        input_pressure = float(input("Enter pressure used to print (psi): "))
        pressure_pa = input_pressure * 6894.76  # Convert psi to Pa
    except ValueError:
        raise ValueError("Invalid pressure input. Please enter a numeric value.")
    
    # Flow Rate
    Vol_flow = calculate_flow_rate(R_In, R_Out, L, K, n, pressure_pa)
    
    # Shear Stress Field
    Z, X, Y, shear_stress = calculate_shear_stress_cross_section(Vol_flow, R_In, R_Out, L, K, n)
    
    # Output
    print("\nNozzle Dimensions:")
    print(f"Inlet Radius (R_0): {R_In*1e6:.0f} μm")
    print(f"Outlet Radius (R_L): {R_Out*1e6:.0f} μm")
    print(f"Length of Tapered Section (L): {L*1e3:.2f} mm")
    print("\nResults:")
    print(f"Consistency Index (K): {K:.4f} Pa·s^n")
    print(f"Power-law Index (n): {n:.4f}")
    print(f"Pressure Input: {input_pressure:.2f} psi ({pressure_pa:.2f} Pa)")
    print(f"Volumetric Flow Rate: {Vol_flow:.6e} m^3/s")
    print(f"Max Shear Stress: {np.max(shear_stress):.4f} Pa (at nozzle outlet)")

    # Visualization
    plot_shear_stress_cross_sections(Z, X, Y, shear_stress, num_plots=5)
