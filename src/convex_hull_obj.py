import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Read the OBJ file and extract vertices
def read_obj_file(file_path):
    vertices = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertex = line.strip().split()[1:]
                vertices.append(list(map(float, vertex)))
    return np.array(vertices)


# Load the OBJ file as a point cloud
obj_file_path = "meshes/Chilli and Capsicum.obj"
point_cloud = read_obj_file(obj_file_path)

# Apply convex hull
hull = ConvexHull(point_cloud)

# Plot the original point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="b", marker="o", label="Point Cloud")

# Plot the convex hull
for simplex in hull.simplices:
    # simplex = np.append(simplex, simplex[0])  # Close the loop
    ax.plot(point_cloud[simplex, 0], point_cloud[simplex, 1], point_cloud[simplex, 2], "r-")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Point Cloud with Convex Hull")
plt.legend()
plt.show()
