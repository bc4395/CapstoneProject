import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from vedo import *

# --- Power-law model ---
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# --- Flow rate ---
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

# --- Compute shear at each mesh point ---
def compute_shear_on_mesh(mesh, Q, R_in, R_out, L, K, n):
    shear_vals = []
    for pt in mesh.points:
        x, y, z = pt
        if 0 <= z <= L:
            r = np.sqrt(x**2 + y**2)
            Rz = R_in - (R_in - R_out) * z / L
            r_norm = min(r, Rz)
            gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r_norm / Rz)**(1/n)
            shear = K * np.abs(gamma_dot)**n
        else:
            shear = 0
        shear_vals.append(shear)
    return shear_vals

# --- Main ---
if __name__ == "__main__":
    df = pd.read_csv("A4C4.csv")
    sr_data = df["SR"].values
    vis_data = df["Vis"].values
    K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

    R_in = 0.00175
    R_out = 0.0004318
    L = 0.0314

    pressure_psi = float(input("Enter pressure used (psi): "))
    pressure_pa = pressure_psi * 6894.76
    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

    mesh = load("conical_nozzle.stl")

    shear_vals = compute_shear_on_mesh(mesh, Q, R_in, R_out, L, K, n)
    mesh.pointdata["ShearStress"] = shear_vals
    mesh.cmap("plasma", "ShearStress").add_scalarbar("Shear Stress (Pa)")

    plt = Plotter(bg='black', axes=1)
    plt.show(mesh, "Shear Stress Colored STL", zoom=1.2)
