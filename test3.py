import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from vedo import *

# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Flow Rate via Numerical Integration -----------------
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    z_vals = np.linspace(0.001, L, 100)
    dz = z_vals[1] - z_vals[0]
    flow_sum = 0
    for z in z_vals:
        Rz = r1 - (r1 - r2) * z / L  # radius decreases from base to tip
        dP_dz = delta_P / L
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)
        flow_sum += Qz * dz
    return flow_sum

# ----------------- Geometry -----------------
R_in = 0.00175     # Base at z = L (largest radius)
R_out = 0.0004318  # Tip at z = 0 (smallest radius)
L = 0.0314

nz = 120           # z-layers
nr = 15            # radial divisions per layer
ntheta = 40        # angular divisions

# ----------------- Fit Viscosity Data -----------------
df = pd.read_csv("A4C4.csv")
sr_data = df["SR"].values
vis_data = df["Vis"].values
K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

# ----------------- Input Pressure and Compute Flow Rate -----------------
pressure_psi = float(input("Enter pressure used (psi): "))
pressure_pa = pressure_psi * 6894.76
Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

# ----------------- Compute Shear Layer-by-Layer -----------------
points = []
shear_vals = []

z_vals = np.linspace(L, 0, nz)  # ← flipped z-direction

for z in z_vals:
    Rz = R_in - (R_in - R_out) * ((L - z) / L)  # radius still shrinks linearly

    # Add center point (r = 0)
    points.append([0.0, 0.0, z])
    shear_vals.append(0.0)  # No shear at center

    r_vals = np.linspace(0, Rz, nr)[1:]  # exclude center
    theta_vals = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

    for r in r_vals:
        for theta in theta_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Compute shear stress
            gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r / Rz)**(1 / n)
            shear = K * np.abs(gamma_dot)**n

            points.append([x, y, z])
            shear_vals.append(shear)

# ----------------- Visualization -----------------
points = np.array(points)
shear_vals = np.array(shear_vals)

try:
    import colorcet
    cmap = colorcet.bmy
except ImportError:
    cmap = "plasma"

cloud = Points(points)
cloud.pointdata["Shear Stress"] = shear_vals
cloud.cmap(cmap, shear_vals, on="points").point_size(3)
cloud.add_scalarbar("Shear Stress (Pa)", c="white")

show(cloud, bg="black", axes=1, title="Shear Stress per Layer in Truncated Cone")
