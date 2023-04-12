import numpy as np
import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("0.ply")
mesh = o3d.io.read_triangle_mesh("sample1.ply")


colors = np.asarray(mesh.vertex_colors)
normals = np.asarray(mesh.vertex_normals)
vertices = np.asarray(mesh.vertices)

print('vertex_colors: ', np.unique(np.asarray(mesh.vertex_colors), axis=0))
print('vertex_normals: ', np.asarray(mesh.vertex_normals)[:5])
print('vertices: ', np.asarray(mesh.vertices)[:5])

#Create a voxel grid from the point cloud with a voxel_size of 0.01

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.03)

# Initialize a visualizer object
vis = o3d.visualization.Visualizer()
# Create a window, name it and scale it
vis.create_window(window_name='Bunny Visualize', width=800, height=600)

# Add the voxel grid to the visualizer
vis.add_geometry(voxel_grid)

# We run the visualizater
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()
