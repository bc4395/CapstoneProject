import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
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
        Rz = r1 - (r1 - r2) * z / L
        dP_dz = delta_P / L
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)
        flow_sum += Qz * dz
    return flow_sum

# ----------------- Shear Stress in Conical Cross-Section -----------------
def compute_cross_section_shear(Q, Rz, K, n, num_xy=300):
    x = np.linspace(-Rz, Rz, num_xy)
    y = np.linspace(-Rz, Rz, num_xy)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R <= Rz

    gamma_dot = np.zeros_like(R)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_dot[mask] = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (R[mask] / Rz)**(1/n)

    shear_stress = np.zeros_like(R)
    shear_stress[mask] = K * np.abs(gamma_dot[mask])**n

    return x, y, shear_stress

# ----------------- Main Script -----------------
if __name__ == "__main__":
    # Load viscosity data and fit power-law model
    try:
        df = pd.read_csv("A4C4.csv")
        sr_data = df['SR'].values
        vis_data = df['Vis'].values
        K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]
    except Exception as e:
        raise RuntimeError(f"Failed to load/fitting data: {e}")

    # Geometry and pressure
    R_in = 0.00175
    R_out = 0.0004318
    L = 0.0314

    try:
        pressure_psi = float(input("Enter pressure used (psi): "))
    except ValueError:
        raise ValueError("Invalid input.")
    pressure_pa = pressure_psi * 6894.76

    # Flow rate (numerically integrated)
    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

    # Sample 3D shear stress across nozzle
    num_z = 15
    num_xy = 300
    z_vals = np.linspace(0, L, num_z)
    shear_stack = []
    x_vals = y_vals = None

    for z in z_vals:
        Rz = R_in - ((R_in - R_out) / L) * z
        x, y, shear_2d = compute_cross_section_shear(Q, Rz, K, n, num_xy)
        shear_stack.append(shear_2d)
        if x_vals is None:
            x_vals = x
            y_vals = y

    shear_volume = np.stack(shear_stack, axis=0)  # shape: (z, y, x)
    flipped_z = L - z_vals  # Match STL orientation

    # Interpolator for shear stress in 3D space
    interpolator = RegularGridInterpolator(
        (flipped_z, y_vals, x_vals),
        shear_volume,
        bounds_error=False,
        fill_value=0
    )

    # Load STL mesh
    mesh = Mesh("conical_nozzle.stl")
    pts = mesh.points
    coords = np.c_[pts[:, 2], pts[:, 1], pts[:, 0]]  # (z, y, x) order
    shear_vals = interpolator(coords)

    try:
        import colorcet
        cmap = colorcet.bmy
    except ImportError:
        cmap = "plasma"

    # Apply colormap and scalar bar to nozzle
    mesh.pointdata["Shear Stress"] = shear_vals
    mesh.cmap(cmap, shear_vals, on="points")
    mesh.alpha(1)
    mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # Slice plane setup
    zmin, zmax = mesh.bounds()[4], mesh.bounds()[5]
    center_z = (zmin + zmax) / 2
    normal = [0, 0, 1]

    vslice = mesh.clone().intersect_with_plane(origin=[0, 0, center_z], normal=normal).triangulate()
    if vslice.npoints > 0:
        pts = vslice.points
        interp_coords = np.c_[np.full(pts.shape[0], center_z), pts[:, 1], pts[:, 0]]
        stress_vals = interpolator(interp_coords)
        vslice.cmap(cmap, stress_vals, on="points").alpha(1)
        vslice.lighting("off")
        vslice.name = "Slice"
        vslice.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # Interactive slicing callback
    def func(w, _):
        c, n = pcutter.origin, pcutter.normal
        zval = c[2]
        slice2d = mesh.clone().intersect_with_plane(origin=c, normal=n).triangulate()
        if not slice2d.npoints:
            return
        pts = slice2d.points
        interp_coords = np.c_[np.full(pts.shape[0], zval), pts[:, 1], pts[:, 0]]
        vals = interpolator(interp_coords)
        slice2d.cmap(cmap, vals, on="points").alpha(1)
        slice2d.name = "Slice"
        slice2d.lighting("off")
        plt.at(1).remove("Slice").add(slice2d)

    # --- Setup vedo plotter ---
    plt = Plotter(N=3, axes=1, bg="k", bg2="bb")

    plt.at(0).add(mesh)

    pcutter = PlaneCutter(vslice, normal=normal, alpha=0, c="white")
    pcutter.add_observer("interaction", func)
    plt.at(1).add(pcutter, vslice, mesh.box())
    pcutter.on()

    plt.show(zoom=1.2)
    plt.interactive()
    plt.close()
