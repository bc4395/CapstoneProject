from vedo import *
import numpy as np

normal = [0, 0, 1]
your_stl_path = "conical_nozzle.stl"

# Try to use colorcet for better color gradients
try:
    import colorcet
    mycmap = colorcet.bmy  # You can change this to any other CC colormap
except ModuleNotFoundError:
    printc("colorcet is not available, using fallback cmap", c='y')
    printc("pip install colorcet", c='y')
    mycmap = ["darkblue", "magenta", (1, 1, 0)]

# Load the STL mesh
mesh = Mesh(your_stl_path).c("orange").lighting("plastic")

# Initial slicing
center = mesh.center_of_mass()
initial_slice = mesh.clone().cut_with_plane(origin=center, normal=normal)
initial_slice.c("lightblue").lw(2).name = "Slice"

# Bounds for slicing limitation (optional)
zmin, zmax = mesh.bounds()[4], mesh.bounds()[5]

# Slicing + color gradient function
def func(w, _):
    c, n = pcutter.origin, pcutter.normal
    zval = c[2]
    if zval < zmin or zval > zmax:
        return  # prevent slicing outside the mesh

    # Slice + fill
    slice2d = mesh.clone().intersect_with_plane(origin=c, normal=n).triangulate()
    
    if slice2d.npoints == 0:
        return  # guard against invalid slice

    # Color using z-values of slice points
    zcoords = slice2d.points[:, 2]
    slice2d.cmap(mycmap, zcoords).alpha(1)
    slice2d.name = "Slice"
    plt.at(1).remove("Slice").add(slice2d)

# Initial filled slice at center
center_z = (zmin + zmax) / 2
vslice = mesh.clone().intersect_with_plane(origin=[0, 0, center_z], normal=normal).triangulate()

if vslice.npoints > 0:
    zcoords = vslice.points[:, 2]
    vslice.cmap(mycmap, zcoords).alpha(1)
vslice.name = "Slice"

# Plotter with 2 views: full + slice
plt = Plotter(axes=0, N=2, bg="k", bg2="bb")

# Add colorbar to the slice (once)
vslice.add_scalarbar(title="z-coordinate", c="w")

# Plane cutter widget
pcutter = PlaneCutter(
    vslice,
    normal=normal,
    alpha=0,
    c="white",
    padding=1,
)
pcutter.add_observer("interaction", func)

# Left: full mesh with docstring
plt.at(0).add(mesh, __doc__)
# Right: just the colored 2D slice
plt.at(1).add(pcutter, mesh.box(), vslice)

pcutter.on()
plt.show(zoom=1.2)
plt.interactive()
plt.close()
