import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from vedo import *

# ----------------- Shear Stress Model Functions -----------------
def model(sr, K, n):
    """Power-law model: eta = K * shear_rate^(n-1)"""
    return K * np.power(sr, n - 1)

def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    if r1 <= 0 or r2 <= 0 or L <= 0 or K <= 0 or n <= 0 or r1 == r2:
        raise ValueError("Invalid input values")
    k = (r1 - r2) / L
    numerator = np.pi**n * n**n * delta_P * k * (r1 * r2)**((3*n + 1)/n)
    denominator = 2 * K * (r1**(3 + 1/n) - r2**(3 + 1/n))
    Q = (numerator / denominator)**(1/n)
    return Q

def calculate_shear_stress_cross_section(Q, R_Init, R_Out, L, K, n, num_z=50, num_xy=100):
    Z = np.linspace(0, L, num_z)
    k = (R_Init - R_Out) / L
    R_z = np.maximum(R_Init - k * Z, 1e-10)

    X = np.zeros((num_xy, num_xy, num_z))
    Y = np.zeros((num_xy, num_xy, num_z))
    shear_stress = np.zeros((num_xy, num_xy, num_z))

    for i, z in enumerate(Z):
        x = np.linspace(-R_z[i], R_z[i], num_xy)
        y = np.linspace(-R_z[i], R_z[i], num_xy)
        X[:, :, i], Y[:, :, i] = np.meshgrid(x, y)
        R = np.sqrt(X[:, :, i]**2 + Y[:, :, i]**2)
        mask = R <= R_z[i]
        shear_rate = np.zeros_like(R)
        shear_rate[mask] = (4 * Q / (np.pi * R_z[i]**3)) * (R[mask] / R_z[i])
        shear_stress[:, :, i][mask] = K * np.abs(shear_rate[mask])**n

    return Z, X, Y, shear_stress

# ----------------- Main Script -----------------
if __name__ == "__main__":
    # Load and fit viscosity data
    try:
        data = pd.read_csv("A4C4.csv")
        sr_data = data['SR'].values
        vis_data = data['Vis'].values
        popt, _ = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])
        K, n = popt
    except Exception as e:
        raise RuntimeError(f"Error loading/fitting data: {e}")

    # Geometry and pressure
    R_In = 0.0015       # meters
    R_Out = 0.0004318   # meters
    L = 0.0314          # meters

    try:
        input_pressure = float(input("Enter pressure used to print (psi): "))
    except ValueError:
        raise ValueError("Invalid input.")
    pressure_pa = input_pressure * 6894.76  # psi to Pa

    # Compute flow rate and shear stress
    Vol_flow = calculate_flow_rate(R_In, R_Out, L, K, n, pressure_pa)
    Z, X, Y, shear_stress = calculate_shear_stress_cross_section(Vol_flow, R_In, R_Out, L, K, n)
    num_z, num_xy = shear_stress.shape[2], shear_stress.shape[0]
    z_vals = np.linspace(0, L, num_z)
    y_vals = np.linspace(-R_In, R_In, num_xy)
    x_vals = np.linspace(-R_In, R_In, num_xy)

    # Interpolator setup for full field
    shear_interp = RegularGridInterpolator(
        (z_vals, y_vals, x_vals),
        shear_stress.transpose(2, 0, 1),  # (z, y, x)
        bounds_error=False,
        fill_value=0
    )

    # ----------------- Vedo Visualization -----------------
    normal = [0, 0, 1]
    stl_path = "conical_nozzle.stl"

    # Colormap
    try:
        import colorcet
        mycmap = colorcet.bmy
    except ImportError:
        printc("colorcet not available, using fallback colormap", c='y')
        mycmap = ["blue", "magenta", "yellow"]

    # Load STL and interpolate stress on mesh
    mesh = Mesh(stl_path)
    mesh_pts = mesh.points
    interp_coords = np.c_[mesh_pts[:, 2], mesh_pts[:, 1], mesh_pts[:, 0]]  # (z, y, x)
    mesh_stress = shear_interp(interp_coords)
    mesh.cmap(mycmap, mesh_stress, on="points")
    mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # Slice plane setup
    zmin, zmax = mesh.bounds()[4], mesh.bounds()[5]
    center_z = (zmin + zmax) / 2
    vslice = mesh.clone().intersect_with_plane(origin=[0, 0, center_z], normal=normal).triangulate()

    if vslice.npoints > 0:
        pts = vslice.points
        interp_coords = np.c_[np.full(pts.shape[0], center_z), pts[:, 1], pts[:, 0]]
        stress_vals = shear_interp(interp_coords)
        vslice.cmap(mycmap, stress_vals, on="points").alpha(1)
        vslice.lighting("off")

    vslice.name = "Slice"
    vslice.add_scalarbar(title="Shear Stress (Pa)", c="w")

    # Interactor callback
    def func(w, _):
        c, n = pcutter.origin, pcutter.normal
        zval = c[2]
        if zval < zmin or zval > zmax:
            return
        slice2d = mesh.clone().intersect_with_plane(origin=c, normal=n).triangulate()
        if slice2d.npoints == 0:
            return
        pts = slice2d.points
        interp_coords = np.c_[np.full(pts.shape[0], zval), pts[:, 1], pts[:, 0]]
        stress_vals = shear_interp(interp_coords)
        slice2d.cmap(mycmap, stress_vals, on="points").alpha(1)
        slice2d.name = "Slice"
        slice2d.lighting("off")
        plt.at(1).remove("Slice").add(slice2d)

    # Plotter layout: left = full mesh, right = slice
    plt = Plotter(axes=1, N=2, bg="k", bg2="bb")
    pcutter = PlaneCutter(vslice, normal=normal, alpha=0, c="white", padding=1)
    pcutter.add_observer("interaction", func)

    plt.at(0).add(mesh, __doc__)

    summary_text = f"""
    Nozzle Geometry:
    Inlet Radius: {R_In * 1e6:.0f} µm
    Outlet Radius: {R_Out * 1e6:.0f} µm
    Length: {L * 1e3:.2f} mm

    Fluid Properties:
    K (Pa·sⁿ): {K:.4f}
    n (Power-law Index): {n:.4f}
    Pressure: {input_pressure:.2f} psi ({pressure_pa:.0f} Pa)
    Flow Rate: {Vol_flow:.3e} m³/s
    """
    # Add the text inside the left plot (full mesh view), bottom-left corner
    text_actor = Text2D(summary_text.strip(),
                        pos='bottom-left',  # position in the window
                        c='w',  # white color
                        bg='k8',  # black background with 80% opacity
                        font='Courier',
                        s=0.8,  # font size scaling
                        justify='left')

    plt.at(0).add(text_actor)

    plt.at(1).add(pcutter, vslice, mesh.box())
    pcutter.on()

    plt.show(zoom=1)
    plt.interactive()
    plt.close()
