import numpy as np
import laspy as lp
import open3d as o3d
import matplotlib.pyplot as plt
import multiprocessing as mp

import roughness_configuration
import math

from tqdm import tqdm
from scipy.linalg import lstsq

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
    self.cloud_path = cloud_path;
    self.point_cloud = lp.read(cloud_path);
    self.points = np.vstack((self.point_cloud.x, self.point_cloud.y,
                       self.point_cloud.z)).transpose();
    self.o3d_point_cloud = o3d.geometry.PointCloud();
    self.o3d_point_cloud.points = o3d.utility.Vector3dVector(self.points);
    return;
  
  def demo(self):
    points = np.vstack((self.point_cloud.x, self.point_cloud.y,
                       self.point_cloud.z)).transpose();
    
    pcd = o3d.geometry.PointCloud();
    pcd.points = o3d.utility.Vector3dVector(points);
    
    o3d.visualization.draw_geometries([pcd]);
    
  def shortest_distance(self, x1, y1, z1, a, b, c, d):
     
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    return d/e
  
  def open3d_octree_test(self):
    """
    IRIgeo = sqrt(((n * sum(di^2)) - (sum(di)^2))/(n * (n - 1)))
    Above is the equation that we will use to calculate the roughness of the pc.
    di is the deviation in elevation of the ith element from the simple linear
    regression for 3.5m around the point.
    
    We are also creating a colour output. Going to have to come up with
    something for this one.
    
    """
    debug = True;
    
    curr_max_depth = 8;
    
    octree = o3d.geometry.Octree(max_depth = curr_max_depth);
    octree.convert_from_point_cloud(self.o3d_point_cloud);
    o3d.visualization.draw_geometries([self.o3d_point_cloud]);
    o3d.visualization.draw_geometries([octree]);
    
        
    return;
    
  def calc_roughness(self):
    """
    IRIgeo = sqrt(((n * sum(di^2)) - (sum(di)^2))/(n * (n - 1)))
    Above is the equation that we will use to calculate the roughness of the pc.
    di is the deviation in elevation of the ith element from the simple linear
    regression for 3.5m around the point.
    
    We are also creating a colour output. Going to have to come up with
    something for this one.
    
    TODO: Convert the for loop into a starmap multiprocessing.
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
      
      # Getting the index of the values that are smaller than the size we are
      # using for plane.
      buffer_idx = np.flatnonzero(bufferout < roughness_configuration.baselength);
      
      # Getting points close to the point being analysed.
      near_points = points[buffer_idx, :];
      
      # Making a best fitting least square plane using the points
      A = near_points.copy();
      B = A.copy()[:, 2];
      A[:, 2] = 1;
      
      # "solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
      fit, residual, rnk, s = lstsq(A, B);
      
      distance = self.shortest_distance(
        points[i, 0], points[i, 1], points[i, 2], fit[0], fit[1], -1, fit[2]
                                        )
      
      # Debugging 1
      if i == 0 and debug:
        print(points);
        print(distance_buffer_array);
        print(bufferout);
        print(buffer_idx);
        print(near_points);
        print(A);
        print(B);
        print(fit);
        print(residual);
        print(rnk);
        print(distance);
        plt.figure();
        ax = plt.subplot(111, projection='3d');
        # "solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
        ax.scatter(near_points[:, 0], near_points[:, 1], near_points[:, 2],
                   color='b');
        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                          np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        ax.plot_wireframe(X,Y,Z, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show();
        
    return;
  
  def calc_roughness_mp(self):
    """
    Multiprocessing version of the calculate roughness algorithm. Should be able
    to complete the calc in less than an hour vs the 22 hours for original.
    """
    debug = True;
    # Making a temporary array for this calculations
    points_list = [];
    for i in range(len(self.point_cloud.x)):
      points_list.append([self.point_cloud.x[i], self.point_cloud.y[i],
                     self.point_cloud.z[i]])
    # Try double for looping to check for point clouds in range.
    
    print("Opening multiprocessing pool.")
    
    mp.freeze_support();
    cores = mp.cpu_count() - 1;
    with mp.Pool(cores) as p:  # Opening up more process pools
      print("Calculating the roughness.");
      # Running parallelization with starmap for significant speed increase.
      para_list = p.starmap(self.roughness_loop,
                                tqdm(
                                  [
                                    (
                                      i, points_list
                                    ) for i in range(len(self.point_cloud.x))
                                  ],
                                    total=len(self.point_cloud.x)))
      print('Completed the processing, closing pools')
      print(para_list);
    p.join();
  
  def roughness_loop(self, i, points_list):
    """
    The process of finding the roughness for each point

    Args:
        i (int): index of the point being viewed
        points (numpy[N, 3] array): The input point cloud's coordinates. Used
        for roughness calculations.

    Returns:
        float: The distance that the point is away from the least square fit
        plane. This represents the roughness of the point.
    """
    points = np.array(points_list);
    distance_buffer_array = np.zeros(points.shape);
    buffer_coord = np.zeros((1, 3));
    # Need to now look for points that are under a certain distance from the
    # point being indexed.
    # This portion of the code is very slow, you need to parallelize it.
    buffer_coord = points[i, :];
    distance_buffer_array = points - buffer_coord;
    bufferout = np.linalg.norm(distance_buffer_array, axis=1);
    
    # Getting the index of the values that are smaller than the size we are
    # using for plane.
    buffer_idx = np.flatnonzero(bufferout < roughness_configuration.baselength);
    
    # Getting points close to the point being analysed.
    near_points = points[buffer_idx, :];
    
    # Making a best fitting least square plane using the points
    A = near_points.copy();
    B = A.copy()[:, 2];
    A[:, 2] = 1;
    
    # "solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
    fit, residual, rnk, s = lstsq(A, B);
    
    distance = self.shortest_distance(
      points[i, 0], points[i, 1], points[i, 2], fit[0], fit[1], -1, fit[2]
                                      )
    return distance;
      