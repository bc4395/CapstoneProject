import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from vedo import *

# --- Power-law model ---
def model(sr, K, n):
    return K * sr ** (n - 1)

# --- Flow rate calculation ---
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

# --- Main ---
if __name__ == "__main__":
    # Load data and fit parameters
    df = pd.read_csv("A4C4.csv")
    sr_data = df["SR"].values
    vis_data = df["Vis"].values
    K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]
    print(f"Fitted K={K:.3e}, n={n:.3f}")

    nozzle = load("conical_nozzle.stl")
    bounds = nozzle.bounds()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    L = zmax - zmin

    R_in = 0.00175
    R_out = 0.0004318

    pressure_psi = float(input("Enter pressure (psi): "))
    pressure_pa = pressure_psi * 6894.76
    Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)
    print(f"Flow rate Q = {Q:.3e}")

    nx, ny, nz = 120, 120, 200
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    R = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    Rz = R_out + (R_in - R_out) * (Z - zmin) / L
    inside = R <= Rz

    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (R / Rz)**(1/n)
        gamma_dot[~inside] = 0
        shear = K * np.abs(gamma_dot)**n
        shear[~inside] = 0

    spacing = [(xmax - xmin)/(nx - 1), (ymax - ymin)/(ny - 1), (zmax - zmin)/(nz - 1)]
    vol = Volume(shear, spacing=spacing).cmap("gist_stern_r").add_scalarbar("Shear Stress (Pa)")

    normal = [0, 0, 1]

    # Initial slice through volume center
    vslice = vol.slice_plane(vol.center(), normal).cmap("gist_stern_r")
    vslice.name = "Slice"

    plt = Plotter(N=2, axes=0, bg="black", bg2="bb")

    def func(w, _):
        c, n = pcutter.origin, pcutter.normal
        new_slice = vol.slice_plane(c, n, autocrop=True).cmap("gist_stern_r")
        new_slice.name = "Slice"
        plt.at(1).remove("Slice").add(new_slice)

    pcutter = PlaneCutter(
        vslice,
        normal=normal,
        alpha=0,
        c="white",
        padding=0,
    )
    pcutter.add_observer("interaction", func)

    plt.at(0).add(nozzle.alpha(0.15), vol, __doc__)
    plt.at(1).add(pcutter, vol.box())
    pcutter.on()

    plt.show(zoom=1.2)
    plt.interactive().close()
