import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Flow Rate via Numerical Integration -----------------
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    z_vals = np.linspace(0.001, L, 100)
    dz = z_vals[1] - z_vals[0]
    flow_sum = 0
    for z in z_vals:
        Rz = r1 - (r1 - r2) * z / L
        dP_dz = delta_P / L
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)
        flow_sum += Qz * dz
    return flow_sum

# ----------------- Generate Shear Stress Grid for 1 Cross Section -----------------
def compute_shear_grid(Q, Rz, K, n, resolution=100):
    r_vals = np.linspace(0, Rz, resolution)
    theta_vals = np.linspace(0, 2 * np.pi, resolution)
    rr, tt = np.meshgrid(r_vals, theta_vals)

    gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (rr / Rz)**(1/n)
    shear = K * np.abs(gamma_dot)**n

    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    return xx, yy, shear

# ----------------- Main Script -----------------
if __name__ == "__main__":
    # Load and fit power-law model
    df = pd.read_csv("A4C4.csv")
    sr_data = df['SR'].values
    vis_data = df['Vis'].values
    K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

    # Geometry and pressure
    R_in = 0.00175
    R_out = 0.0004318
    L = 0.0314

    pressure_psi = float(input("Enter pressure used (psi): "))
    pressure_pa = pressure_psi * 6894.76
    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

    # Set up subplots
    num_sections = 15
    z_vals = np.linspace(0, L, num_sections)
    ncols = 5
    nrows = int(np.ceil(num_sections / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), constrained_layout=True)

    for i, z in enumerate(z_vals):
        Rz = R_in - ((R_in - R_out) / L) * z
        xx, yy, shear = compute_shear_grid(Q, Rz, K, n)

        row, col = divmod(i, ncols)
        ax = axs[row, col] if nrows > 1 else axs[col]
        pcm = ax.pcolormesh(xx, yy, shear, shading='auto', cmap='plasma')
        ax.set_title(f"Z = {z*1000:.1f} mm")
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i+1, nrows*ncols):
        row, col = divmod(j, ncols)
        ax = axs[row, col] if nrows > 1 else axs[col]
        ax.axis('off')

    # Add colorbar
    fig.colorbar(pcm, ax=axs, orientation='vertical', shrink=0.7, label='Shear Stress (Pa)')
    plt.suptitle("Shear Stress Cross Sections (Filled)", fontsize=16)
    plt.show()
