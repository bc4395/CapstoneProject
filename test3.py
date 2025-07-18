# IGNORE_FILE


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from vedo import *

# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Shear Stress Calculation -----------------
def compute_shear_stress(r, Rz, Q, K, n):
    if r == 0:
        return 0.0
    gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r / Rz)**(1 / n)
    return K * np.abs(gamma_dot)**n

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

# ----------------- Geometry & Grid Setup -----------------
R_in = 0.00175     # Base at z = L (m)
R_out = 0.0004318  # Tip at z = 0 (m)
L = 0.0314         # Length (m)

nz = 101
nr = 50
ntheta = 240

z_vals = np.linspace(L, 0, nz)
y_vals = np.linspace(-R_in, R_in, 2*nr)
x_vals = np.linspace(-R_in, R_in, 2*nr)

X, Y = np.meshgrid(x_vals, y_vals)

# ----------------- Fit Viscosity Data -----------------
df = pd.read_csv("A4C4.csv")
sr_data = df["SR"].values
vis_data = df["Vis"].values
K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

# ----------------- Pressure Input & Flow -----------------
pressure_psi = float(input("Enter pressure used (psi): "))
pressure_pa = pressure_psi * 6894.76
Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

# ----------------- Compute 3D Shear Volume -----------------
shear_volume = np.zeros((nz, len(y_vals), len(x_vals)))
for i, z in enumerate(z_vals):
    Rz = R_in - (R_in - R_out) * ((L - z) / L)
    for j, y in enumerate(y_vals):
        for k, x in enumerate(x_vals):
            r = np.sqrt(x**2 + y**2)
            shear_volume[i, j, k] = compute_shear_stress(r, Rz, Q, K, n)

# ----------------- Interpolator -----------------
flipped_z_vals = z_vals[::-1]
interpolator = RegularGridInterpolator(
    (flipped_z_vals, y_vals, x_vals),
    shear_volume[::-1],
    bounds_error=False,
    method='linear',
    fill_value=None
)

# ----------------- Load STL Nozzle -----------------
mesh = Mesh("conical_nozzle.stl")
pts = mesh.points
coords = np.c_[pts[:, 2], pts[:, 1], pts[:, 0]]  # (z, y, x)
shear_vals = interpolator(coords)

# ----------------- Apply Shear Stress Colors to 3D Mesh -----------------
try:
    import colorcet
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bmy", colorcet.bmy)
except ImportError:
    cmap = "plasma"

shear_min = np.nanmin(shear_volume)
shear_max = np.nanmax(shear_volume)

mesh.pointdata["Shear Stress"] = shear_vals
mesh.cmap(cmap, shear_vals, on="points")
mesh.alpha(1)
mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

# ----------------- Show 3D Model -----------------

plt1 = Plotter(title="3D Shear Field", size=(900, 700), axes=1, bg="k")
plt1.show(mesh, zoom=1.2, viewup="z", interactive=True)
plt1.close()

# ----------------- Cross Sectional View -----------------
#import matplotlib.pyplot as plt

c_section_height = float(input("Enter cross-section height as percentage of length (0-100): "))
z2 = L * (c_section_height / 100.0)

# Arrays to store cross-section locaiton and shear values
cross_pts = []
cross_xy = {}
cross_z = []
cross_shear_val = []
cross_shear_pts = {}

# Calculating radius values at the specified height
Rz2 = R_in - (R_in - R_out) * ((L - z2) / L)
cross_pts.append([0.0, 0.0, z2])
cross_shear_val.append(0.0)

r_vals = np.linspace(0, Rz2, nr)[1:]
theta_vals = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

for r in r_vals:
    for theta in theta_vals:
        x2 = r * np.cos(theta)
        y2 = r * np.sin(theta)
        shear = compute_shear_stress(r, Rz2, Q, K, n)
        cross_pts.append([x2, y2, z2])
        cross_xy[z2].append([x2, y2])
        cross_shear_pts[(z2, x2, y2)] = shear
        cross_shear_val.append(shear)