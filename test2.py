import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from vedo import *

# ----------- Power-law viscosity model --------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------- Flow rate calc -------------------------
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

# -------------- Main ---------------------------------
if __name__ == "__main__":
    # Load and fit viscosity data
    df = pd.read_csv("A4C4.csv")
    sr_data = df["SR"].values
    vis_data = df["Vis"].values
    K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

    # Geometry
    R_in = 0.00175
    R_out = 0.0004318
    L = 0.0314

    pressure_psi = float(input("Enter pressure used (psi): "))
    pressure_pa = pressure_psi * 6894.76

    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

    # Compute shear points cloud
    nz, nr, ntheta = 120, 15, 40

    points = []
    shear_vals = []

    z_vals = np.linspace(L, 0, nz)
    for z in z_vals:
        Rz = R_in - (R_in - R_out) * ((L - z) / L)
        points.append([0,0,z])
        shear_vals.append(0)

        r_vals = np.linspace(0, Rz, nr)[1:]
        theta_vals = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

        for r in r_vals:
            for theta in theta_vals:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                gamma_dot = ((n+1)/n) * (2*Q / (np.pi*Rz**3)) * (r/Rz)**(1/n)
                shear = K * np.abs(gamma_dot)**n
                points.append([x,y,z])
                shear_vals.append(shear)

    points = np.array(points)
    shear_vals = np.array(shear_vals)

    # Load STL mesh
    mesh = Mesh("conical_nozzle.stl")
    mesh_pts = mesh.points

    # ----------- Interpolate shear on mesh using KDTree for nearest neighbors -----------
    # Build KDTree from shear points cloud
    tree = cKDTree(points)
    dist, idx = tree.query(mesh_pts, k=5)  # 5 nearest neighbors

    # Weight inversely by distance (avoid zero division)
    weights = 1 / (dist + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average shear stress for mesh points
    shear_on_mesh = np.sum(shear_vals[idx] * weights, axis=1)

    mesh.pointdata["Shear Stress"] = shear_on_mesh

    # --------- Create a volume grid inside the nozzle bbox -------------
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds()
    nx, ny, nz = 50, 50, 100  # grid resolution

    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    zg = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Use griddata to interpolate shear stress on volume grid
    # 'linear' interpolation, fill points outside convex hull with 0
    grid_shear = griddata(points, shear_vals, grid_points, method='linear', fill_value=0)
    grid_shear = grid_shear.reshape((nx, ny, nz))

    # -------- Visualization -------------------
    try:
        import colorcet
        cmap = colorcet.bmy
    except ImportError:
        cmap = "plasma"

    mesh.cmap(cmap, shear_on_mesh, on="points").alpha(1)
    mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # Create Volume object from interpolated shear volume
    spacing = [xg[1]-xg[0], yg[1]-yg[0], zg[1]-zg[0]]
    vol = Volume(grid_shear, spacing=spacing).cmap(cmap).add_scalarbar("Shear Stress (Pa)")

    # Setup slicing plane through volume
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    normal = [0,0,1]
    vslice = vol.slice_plane(origin=center, normal=normal).cmap(cmap)
    vslice.name = "Slice"

    # Interactive slicer
    pcutter = PlaneCutter(vslice, normal=normal, alpha=0, c="white")

    def on_slice_interaction(widget, _):
        c, n = pcutter.origin, pcutter.normal
        slice2d = vol.slice_plane(origin=c, normal=n, autocrop=True).cmap(cmap)
        slice2d.name = "Slice"
        plt.at(1).remove("Slice").add(slice2d)

    pcutter.add_observer("interaction", on_slice_interaction)

    # Setup plotter with 2 panels: mesh and volume+slice
    plt = Plotter(N=2, axes=1, bg="black", bg2="bb")
    plt.at(0).add(mesh)
    plt.at(1).add(pcutter, vol.box(), vslice)
    pcutter.on()

    plt.show(zoom=1.2)
    plt.interactive()
    plt.close()
