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
R_in = 0.00175     # Base at z = L
R_out = 0.0004318  # Tip at z = 0
L = 0.0314

nz = 100           # axial layers
nr = 45            # radial points
ntheta = 240        # angular slices

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
    shear_volume[::-1],  # match z-flip
    bounds_error=False,
    method='linear',
    fill_value=None
)

# ----------------- Load STL & Apply Shear -----------------
mesh = Mesh("conical_nozzle.stl")
pts = mesh.points
coords = np.c_[pts[:, 2], pts[:, 0], pts[:, 1]]  # (z, y, x)
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

# ----------------- Initial Slice -----------------
zmin, zmax = mesh.bounds()[4], mesh.bounds()[5]
center_z = (zmin + zmax) / 2
normal = [0, 0, 1]

vslice = mesh.slice(normal=normal, origin=[0, 0, center_z])
vslice.name = "Slice"

# ----------------- Interactive Slice Handler -----------------
def func(w, _):
    c, n = pcutter.origin, pcutter.normal
    zval = c[2]
    slice2d = mesh.slice(normal=n, origin=c)
    if not slice2d.npoints:
        return
    pts = slice2d.points
    interp_coords = np.c_[np.full(pts.shape[0], zval), pts[:, 1], pts[:, 0]]
    vals = interpolator(interp_coords)
    slice2d.cmap(cmap, vals, on="points").alpha(1)
    slice2d.name = "Slice"
    slice2d.lighting("off")
    vslice.add_scalarbar(title="Shear Stress (Pa)", c="w")
    plt.at(1).remove("Slice").add(slice2d)

# ----------------- Original Point Cloud -----------------
points = []
shear_vals_pts = []

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
            shear_vals_pts.append(shear)

cloud = Points(np.array(points))
cloud.pointdata["Shear Stress"] = np.array(shear_vals_pts)
cloud.cmap(cmap, shear_vals_pts, on="points").point_size(3)
cloud.add_scalarbar("Shear Stress (Pa)", c="white")

# ----------------- Final Visualization -----------------
plt = Plotter(N=2, axes=1, bg="k", bg2="bb")
plt.at(0).add(mesh)

pcutter = PlaneCutter(vslice, normal=normal, alpha=0, c="white", padding=0,)
pcutter.add_observer("interaction", func)
plt.at(1).add(pcutter, vslice, mesh.box())
pcutter.on()

plt.show(zoom=1.2)
plt.interactive()
plt.close()
