import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from vedo import *
from tqdm import tqdm  # <-- progress bar

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

# ----------------- Main -----------------
if __name__ == "__main__":
    # --- Load viscosity data and fit power-law model ---
    try:
        df = pd.read_csv("A4C4.csv")
        sr_data = df['SR'].values
        vis_data = df['Vis'].values
        K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]
    except Exception as e:
        raise RuntimeError(f"Failed to load or fit data: {e}")

    # --- Geometry and pressure ---
    R_in = 0.00175     # Inlet radius (m)
    R_out = 0.0004318  # Outlet radius (m)
    L = 0.0314         # Nozzle length (m)

    try:
        pressure_psi = float(input("Enter pressure used (psi): "))
    except ValueError:
        raise ValueError("Invalid input.")
    pressure_pa = pressure_psi * 6894.76

    # --- Compute flow rate ---
    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

    # --- Generate point cloud inside conical nozzle ---
    nz = 120       # z-layers
    nr = 40        # radial divisions
    ntheta = 80    # angular steps

    z_vals = np.linspace(0, L, nz)
    points = []
    shear_vals = []

    print("Generating shear stress points...")
    for z in tqdm(z_vals, desc="Generating points"):
        Rz = R_in - (R_in - R_out) * (z / L)
        r_vals = np.linspace(0, Rz, nr)
        theta_vals = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

        for r in r_vals:
            for theta in theta_vals:
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                with np.errstate(divide='ignore', invalid='ignore'):
                    gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r / Rz)**(1 / n)
                    shear = K * np.abs(gamma_dot)**n

                points.append([x, y, z])
                shear_vals.append(shear)

    points = np.array(points)
    shear_vals = np.array(shear_vals)

    # --- Create interpolator for shear stress ---
    coords = np.c_[points[:, 2], points[:, 1], points[:, 0]]  # (z, y, x)
    interpolator = LinearNDInterpolator(coords, shear_vals, fill_value=0)

    # --- Load STL mesh and map shear stress ---
    mesh = Mesh("conical_nozzle.stl")
    mesh_pts = mesh.points
    mesh_coords = np.c_[mesh_pts[:, 2], mesh_pts[:, 1], mesh_pts[:, 0]]
    shear_on_mesh = interpolator(mesh_coords)

    try:
        import colorcet
        cmap = colorcet.bmy
    except ImportError:
        cmap = "plasma"

    mesh.pointdata["Shear Stress"] = shear_on_mesh
    mesh.cmap(cmap, shear_on_mesh, on="points")
    mesh.alpha(1)
    mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # --- Build volumetric shear stress grid for interactive slicing ---
    x_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    y_vals = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    z_vals_grid = np.linspace(points[:, 2].min(), points[:, 2].max(), 200)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals_grid, indexing='ij')
    grid_coords = np.c_[Z.ravel(), Y.ravel(), X.ravel()]

    print("Interpolating shear stress over volume grid...")
    shear_vals_grid = np.empty(len(grid_coords))
    chunk_size = 100000

    for i in tqdm(range(0, len(grid_coords), chunk_size), desc="Interpolating volume grid"):
        shear_vals_grid[i:i+chunk_size] = interpolator(grid_coords[i:i+chunk_size])

    grid_shear = shear_vals_grid.reshape(X.shape)

    spacing = [x_vals[1] - x_vals[0], y_vals[1] - y_vals[0], z_vals_grid[1] - z_vals_grid[0]]
    vol = Volume(grid_shear, spacing=spacing).cmap(cmap).add_scalarbar("Shear Stress (Pa)")

    # --- Setup interactive slice ---
    normal = [0, 0, 1]
    initial_center = [0, 0, L / 2]
    vslice = vol.slice_plane(origin=initial_center, normal=normal).cmap(cmap)
    vslice.name = "Slice"

    pcutter = PlaneCutter(
        vslice,
        normal=normal,
        alpha=0,
        c="white",
        padding=0,
    )

    def func(w, _):
        c, n = pcutter.origin, pcutter.normal
        sliced = vol.slice_plane(c, n, autocrop=True).cmap(cmap)
        sliced.name = "Slice"
        plt.at(1).remove("Slice").add(sliced)

    pcutter.add_observer("interaction", func)

    # --- Create 3D point cloud (optional view) ---
    point_cloud = Points(points)
    point_cloud.pointdata["Shear Stress"] = shear_vals
    point_cloud.cmap(cmap, shear_vals, on="points").point_size(3)

    # --- Plot everything ---
    plt = Plotter(N=2, axes=1, bg="k", bg2="bb")
    plt.at(0).add(mesh, point_cloud)
    plt.at(1).add(pcutter, vol.box(), vslice)
    pcutter.on()
    plt.show(zoom=1.2)
    plt.interactive().close()
