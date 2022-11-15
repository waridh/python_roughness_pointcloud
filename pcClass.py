import numpy as np
import laspy as lp
import open3d as o3d

from tqdm import tqdm

class PointCloudCST():
  """
  This class will be used to store the point cloud file. This way, we keep
  things encapsulated. Roughness calculations will be done through here.
  """
  def __init__(self, cloud_path):
    """
    Initialization includes taking in the file address so that we can hold the
    point cloud data immediately. Only works on one file at a time currently.

    Args:
        cloud_path (string): Address of the file we are reading.
    """
    self.point_cloud = lp.read(cloud_path);
    return;
  
  def demo(self):
    points = np.vstack((self.point_cloud.x, self.point_cloud.y,
                       self.point_cloud.z)).transpose();
    
    pcd = o3d.geometry.PointCloud();
    pcd.points = o3d.utility.Vector3dVector(points);
    
    o3d.visualization.draw_geometries([pcd]);
    
  def calc_roughness(self):
    """
    IRIgeo = sqrt(((n * sum(di^2)) - (sum(di)^2))/(n * (n - 1)))
    Above is the equation that we will use to calculate the roughness of the pc.
    di is the deviation in elevation of the ith element from the simple linear
    regression for 3.5m around the point.
    
    We are also creating a colour output. Going to have to come up with
    something for this one.
    """
    debug = True;
    # Making a temporary array for this calculations
    points = np.vstack((self.point_cloud.x, self.point_cloud.y,
                       self.point_cloud.z)).transpose();
    distance_buffer_array = np.zeros(points.shape);
    buffer_coord = np.zeros((1, 3));
    # Try double for looping to check for point clouds in range.
    for i in tqdm(range(points.shape[0])):
      # Need to now look for points that are under a certain distance from the
      # point being indexed.
      # This portion of the code is very slow, you need to parallelize it.
      buffer_coord = points[i, :];
      distance_buffer_array = points - buffer_coord;
      bufferout = np.linalg.norm(distance_buffer_array, axis=1);
      
      # Debugging 1
      if i == 0 and debug:
        print(points);
        print(distance_buffer_array);
        print(bufferout);
        
    return;