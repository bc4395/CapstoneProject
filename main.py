import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
ntheta = 360

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
cloud_pts = {}
for i, z in enumerate(z_vals):
    Rz = R_in - (R_in - R_out) * ((L - z) / L)
    for j, y in enumerate(y_vals):
        for k, x in enumerate(x_vals):
            r = np.sqrt(x**2 + y**2)
            shear_volume[i, j, k] = compute_shear_stress(r, Rz, Q, K, n)
            cloud_pts[(z, y, x)] = shear_volume[i, j, k]

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
# Arrays to store cross-section locaiton and shear values
points = []
shear_vals_pts = []
xy_points = {}
shear_points = {}

# Initialize xy_points dictionary
for z in z_vals:
    if z not in xy_points:
        xy_points[z] = []

# Calculate shear stress at all grid points for cross sections
for z in z_vals:
    Rz = R_in - (R_in - R_out) * ((L - z) / L)
    points.append([0.0, 0.0, z])
    shear_vals_pts.append(0.0)

    r_vals = np.linspace(0, Rz, nr)[1:]
    theta_vals = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

    for r in r_vals:
        for theta in theta_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            shear = compute_shear_stress(r, Rz, Q, K, n)
            points.append([x, y, z])
            xy_points[z].append([x, y])
            shear_points[(z, x, y)] = shear
            shear_vals_pts.append(shear)

# Determine global min/max shear for consistent color scaling
min_shear = min(shear_vals_pts)
max_shear = max(shear_vals_pts)

# --- Setup custom subplot layout using GridSpec ---
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig)

# Place plots: top-left, top-right, bottom-left
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0])
]

# --- Recalculating for Cross sections ---
percentages = []
for i in range(3):
    percentage = float(input(f"Enter cross section height {i+1} as percentage (0-100): "))
    percentages.append(percentage)
    slice_z = L * (percentage / 100.0)

    if slice_z not in xy_points:
        xy_points[slice_z] = []
        Rz = R_in - (R_in - R_out) * ((L - slice_z) / L)
        points.append([0.0, 0.0, slice_z])
        shear_vals_pts.append(0.0)

        r_vals = np.linspace(0, Rz, nr)[1:]
        theta_vals = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

        for r in r_vals:
            for theta in theta_vals:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                shear = compute_shear_stress(r, Rz, Q, K, n)
                points.append([x, y, slice_z])
                xy_points[slice_z].append([x, y])
                shear_points[(slice_z, x, y)] = shear
                shear_vals_pts.append(shear)

# --- Plot each subplot ---
for ax, percentage in zip(axes, percentages):
    slice_z = L * (percentage / 100.0)
    Rz = R_in - (R_in - R_out) * ((L - slice_z) / L)

    slice_points = xy_points[slice_z]
    slice_shear_vals = []

    for pt in slice_points:
        x, y = pt[0], pt[1]
        shear = shear_points.get((slice_z, x, y), 0.0)
        slice_shear_vals.append(shear)

    sc = ax.scatter(*zip(*slice_points), c=slice_shear_vals, cmap='plasma', vmin=min_shear, vmax=max_shear)
    ax.set_title(f'Shear Stress at {percentage:.1f}% (z = {slice_z:.2f} m, R = {Rz:.4f} m)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.grid(False)

# --- Show Plots ---
cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.3])
fig.colorbar(sc, cax=cbar_ax, label='Shear Stress (Pa)')
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()