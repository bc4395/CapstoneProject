import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Flow Rate Calculation -----------------
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    k = (r1 - r2) / L
    numerator = np.pi**n * n**n * delta_P * k * (r1 * r2)**((3*n + 1)/n)
    denominator = 2 * K * (r1**(3 + 1/n) - r2**(3 + 1/n))
    return (numerator / denominator)**(1/n)

# ----------------- Compute Shear Stress on Radial Grid -----------------
def compute_shear_stress_radial(z, Q, R1, R2, L, K, n, num_r=120, num_theta=360):
    k = (R1 - R2) / L
    Rz = max(R1 - k * z, 1e-10)

    r = np.linspace(0, Rz, num_r)
    theta = np.linspace(0, 2 * np.pi, num_theta)
    R_grid, Theta = np.meshgrid(r, theta)

    # Fixed shear rate: zero at center, max at wall (correct physical profile)
    gamma_dot_r = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r / Rz)**(1/n)

    # Expand gamma_dot_r along theta axis to create 2D array
    gamma_dot = np.tile(gamma_dot_r, (num_theta, 1))

    # Shear stress from power-law model
    shear_stress = K * np.abs(gamma_dot)**n

    # Convert polar to cartesian for plotting
    X = R_grid * np.cos(Theta)
    Y = R_grid * np.sin(Theta)

    return X, Y, shear_stress

# ----------------- Main Program -----------------
if __name__ == "__main__":
    # Load viscosity data and fit model
    try:
        data = pd.read_csv("A4C4.csv")
        sr_data = data['SR'].values
        vis_data = data['Vis'].values
        K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]
    except Exception as e:
        raise RuntimeError(f"Error loading/fitting viscosity data: {e}")

    # Nozzle geometry parameters (meters)
    R_In = 0.00175
    R_Out = 0.0004318
    L = 0.0314

    # Input pressure in psi to Pa
    try:
        P_input = float(input("Enter pressure in psi: "))
    except ValueError:
        raise ValueError("Invalid pressure input.")
    delta_P = P_input * 6894.76

    # Calculate flow rate Q
    Q = calculate_flow_rate(R_In, R_Out, L, K, n, delta_P)

    # Choose cross-section location z (meters)
    z = L / 2

    # Compute shear stress distribution in cross-section
    X, Y, shear_stress = compute_shear_stress_radial(z, Q, R_In, R_Out, L, K, n)

    # Plot shear stress (Pa) over circular cross-section
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X * 1e3, Y * 1e3, shear_stress, levels=100, cmap='plasma')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title(f"Shear Stress [Pa] at z = {z*1e3:.2f} mm")
    cbar = plt.colorbar(contour)
    cbar.set_label("Shear Stress [Pa]")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
