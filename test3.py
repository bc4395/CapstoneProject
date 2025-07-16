import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
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
R_in = 0.00175     # Base at z = L
R_out = 0.0004318  # Tip at z = 0
L = 0.0314

nz = 250
nr = 50

z_vals = np.linspace(L, 0, nz)
y_vals = np.linspace(-R_in, R_in, 2*nr)
x_vals = np.linspace(-R_in, R_in, 2*nr)

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

# ----------------- Load STL & Apply Shear -----------------
mesh = Mesh("conical_nozzle.stl")
pts = mesh.points
coords = np.c_[pts[:, 2], pts[:, 1], pts[:, 0]]  # (z, y, x)
shear_vals = interpolator(coords)

try:
    import colorcet
    cmap = colorcet.bmy
except ImportError:
    cmap = "plasma"

mesh.pointdata["Shear Stress"] = shear_vals
mesh.cmap(cmap, shear_vals, on="points")
mesh.alpha(1)
mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

# ----------------- Load and Position random.stl -----------------
cell = Mesh("random.stl")

# Shift to center inside the nozzle (e.g., z = L/2)
target_center = np.array([0.0003, 0.0003, L / 2])
cell_center = cell.center_of_mass()
cell.shift(target_center - cell_center)

# ----------------- Nearest Neighbor Mapping: Transfer Colors Instead of Values -----------------
nozzle_tree = cKDTree(mesh.points)
_, idx = nozzle_tree.query(cell.points)  # nearest nozzle point for each random.stl point

# Transfer both stress values and actual RGB colors
transferred_shear = mesh.pointdata["Shear Stress"][idx]
transferred_colors = mesh.pointcolors[idx]

# Apply to random.stl
cell.pointdata["Shear Stress"] = transferred_shear
cell.pointcolors = transferred_colors
cell.alpha(1)

# ----------------- Show Main Nozzle View (plt1) -----------------
plt1 = Plotter(title="Shear Field View", size=(900, 700), axes=1, bg="k")
plt1.show(mesh, zoom=1.2, viewup="z", interactive=False)

# ----------------- Show Random STL as Simulation Cell (plt2) -----------------
plt2 = Plotter(title="Simulation Cell View (random.stl)", size=(600, 600), axes=1, bg="bb")
plt2.show(cell, zoom=1.5, viewup="z", interactive=True)
