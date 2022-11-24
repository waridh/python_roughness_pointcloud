import numpy as np
import scipy as sp
import laspy as lp
import open3d as o3d
import matplotlib.pyplot as plt
import multiprocessing as mp

import roughness_configuration
import math

from tqdm import tqdm
from scipy.linalg import lstsq

def roughness2_loop(i, points, lstsqrplnrad, ckd_tree):
  """
  The loop code, in which we are finding the roughness value of one point.

  Args:
      i (int): The index of the point that is being analysed right now.
  """
  
  buffer_coord = np.zeros((1, 3));
  # Need to now look for points that are under a certain distance from the
  # point being indexed.
  # This portion of the code is very slow, you need to parallelize it.
  
  # Getting points close to the point being analysed.
  idx = ckd_tree.query_ball_point(points[i, :], lstsqrplnrad);
  
  # Making a best fitting least square plane using the points
  A = points[idx[:], :].copy();
  B = A.copy()[:, 2];
  A[:, 2] = 1;
  
  # "solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
  fit, residual, rnk, s = lstsq(A, B);
  
  distance = shortest_distance(
    points[i, 0], points[i, 1], points[i, 2], fit[0], fit[1], -1, fit[2]
                                    )
  
  return distance;

def shortest_distance(x1, y1, z1, a, b, c, d):
    """
    Simply calculates the shortest distance between a point and a hyperplane.

    Args:
        x1 (float): The x coordinate of the point
        y1 (_float_): The y coordinate of the point
        z1 (_float_): The z coordinate of the point
        a (_float_): The coefficient of the plane that belongs to x.
        b (_float_): The coefficient of the plane that belongs to y.
        c (_float_): The coefficient of the plane that belongs to z.
        d (_float_): The hanging coefficient of the plane equation.

    Returns:
        (_float_): The distance from the point decribed in the input to the
        plane given in the input.
    """
     
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    return d/e

class PointCloudCST():
  """
  This class will be used to store the point cloud file. This way, we keep
  things encapsulated. Roughness calculations will be done through here.
  """
  def __init__(self, cloud_path, lstsqrplnrad):
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
    self.lstsqrplnrad = lstsqrplnrad;
    
    return;
  
  def print2las(self, output_path, mode="roughness"):
    """
    This method prints the current cloud contained into a .las file format. The
    mode added will determine which process would get added to that output.

    Args:
        output_path (str): The path of the output file. Needs to be a string,
        and should be absolute.
        mode (str, optional): The process that is being outputted. Defaults to
        "roughness".
    """
    rgbinfo = (255,0,0)
    red, green, blue = rgbinfo       
    classify = 10
              
    point_count = self.points.shape[0]
    print(self.points.shape)            
    print(f'Main las file points = {point_count}')
    filler = np.empty((point_count,1), dtype = int)
    if mode == "roughness":
      
      pointrecord = lp.create(file_version="1.2", point_format=3)
      pointrecord.header.offsets = np.min(self.points, axis=0)
      pointrecord.header.scales = [0.001, 0.001, 0.001]
      pointrecord.header.generating_software = "SSI_RoadScan"
      pointrecord.header.point_count = point_count
      pointrecord = lp.create(point_format=3,file_version="1.2") 
      pointrecord.x = self.points[:,0];
      pointrecord.y = self.points[:,1];
      pointrecord.z = self.points[:,2];
      filler.fill(classify)
      pointrecord.classification = filler[:,0]
      
      d_max = max(self.roughness_array);
      d_min = min(self.roughness_array);
      d_ave = ((d_max+d_min)/2);
      filler.fill(red)
      pointrecord.red = filler[:, 0]
      filler.fill(blue)
      pointrecord.blue = filler[:,0]
      filler.fill(green)
      pointrecord.green = self.point_cloud.intensity;
      pointrecord.intensity = self.roughness_array / d_ave * 127;
      pointrecord.write(output_path);
      return;
  
  def demo(self):
    points = np.vstack((self.point_cloud.x, self.point_cloud.y,
                       self.point_cloud.z)).transpose();
    
    pcd = o3d.geometry.PointCloud();
    pcd.points = o3d.utility.Vector3dVector(points);
    
    o3d.visualization.draw_geometries([pcd]);
    
  
  
  def downsampler(self):
    """
    When we need to downsample the code.
    """
    self.o3d_point_cloud = self.o3d_point_cloud.voxel_down_sample(
      voxel_size=(self.lstsqrplnrad/3)
      );
    self.points = np.asarray(self.o3d_point_cloud.points);
    
  def make_kdtrees(self):
    """
    Create a kd tree when needed
    """
    self.pcd_tree = o3d.geometry.KDTreeFlann(self.o3d_point_cloud);
    
  def make_ckdtrees(self):
    """Creates the supperrior ckd tree.
    """
    self.ckd_tree = sp.spatial.cKDTree(self.points);
    
  def visualize_kdtree(self):
    """
    Just used to visualize the input point cloud to confirm that everything is
    working as it should. Not important to roughness yet.
    """
    # Initialize a visualizer object
    vis = o3d.visualization.Visualizer()
    # Create a window, name it and scale it
    vis.create_window(window_name='Cloud Visualize', width=800, height=600)

    # Set background color to black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # Add the point cloud to the visualizer
    vis.add_geometry(self.o3d_point_cloud)

    # Run the visualizater
    vis.run()
    # Once the visualizer is closed destroy the window and clean up
    vis.destroy_window()
  
  def open3d_octree_test(self):
    """
    IRIgeo = sqrt(((n * sum(di^2)) - (sum(di)^2))/(n * (n - 1)))
    Above is the equation that we will use to calculate the roughness of the pc.
    di is the deviation in elevation of the ith element from the simple linear
    regression for 3.5m around the point.
    
    We are also creating a colour output. Going to have to come up with
    something for this one.
    
    some powerful functions. o3d.geometry.KDTreeFlann(name_of_3d_object);
    search_radius_vector_3d will then search for all neighbours within a certain
    radius.
    
    This code doesn't really do anything other than testing for me.
    
    """
    debug = True;
    
    self.downsampler();
    
    self.make_kdtrees();
    
    self.visualize_kdtree();
    
    [k, idx, _] = self.pcd_tree.search_radius_vector_3d(self.points[0, :], 3);
    print(idx);
    output = self.points[idx[:], :];
    print(output);
    
        
    return;
  
    
  def roughness2(self, debug=False):
    
    """
    This is the main program that will run roughness analysis on the point cloud
    contained in this object. It will create a roughness output that describes
    the distance between each point and the best fit least sqaure plane around
    the point in the radius given at the input stage.
    
    Output:
      (Nx1 numpy array): The roughness array described earlier.
    """
    
    #self.downsampler();
    
    self.make_ckdtrees();
    
    self.roughness_array = np.zeros(self.points.shape[0]);
    
    # for i in tqdm(range(self.points.shape[0])):
      
    #   self.roughness2_loop(i);
      
    #   if (i == 30) and debug:
    #     print(self.roughness_array[i]);
        
        
    mp.freeze_support();
    cores = mp.cpu_count() - 1;
    with mp.Pool(cores) as p:  # Opening up more process pools
      print("Calculating the roughness.");
      # Running parallelization with starmap for significant speed increase.
      rough_buffer = p.starmap(roughness2_loop,
                              tqdm(
                                [
                                  (
                                    i, self.points, self.lstsqrplnrad,
                                    self.ckd_tree
                                  ) for i in range(self.points.shape[0])
                                ],
                                   total=self.points.shape[0]))
    print('Completed the processing, closing pools')
    p.join();
        
    self.roughness_array = np.array(rough_buffer);
  
  
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
    buffer_idx = np.flatnonzero(bufferout < self.lstsqrplnrad);
    
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
      
# Params class containing the counter and the main LineSet object holding all line subsets
# class params():

#   counter = 0
#   full_line_set = o3d.geometry.LineSet()

#   # Callback function for generating the LineSets from each neighborhood
#   def build_edges(vis):
#     # Run this part for each point in the point cloud
#     if params.counter < len(points):
#       # Find the K-nearest neighbors for the current point. In our case we use 6
#       [k, idx, _] = pcd_tree.search_knn_vector_3d(points[params.counter,:], 6)
#       # Get the neighbor points from the indices
#       points_temp = points[idx,:]
      
#       # Create the neighbours indices for the edge array
#       neighbours_num = np.arange(len(points_temp))
#       # Create a temp array for the center point indices
#       point_temp_num = np.zeros(len(points_temp))
#       # Create the edges array as a stack from the current point index array and the neighbor indices array
#       edges = np.vstack((point_temp_num,neighbours_num)).T

#       # Create a LineSet object and give it the points as nodes together with the edges
#       line_set = o3d.geometry.LineSet()
#       line_set.points = o3d.utility.Vector3dVector(points_temp)
#       line_set.lines = o3d.utility.Vector2iVector(edges)
#       # Color the lines by either using red color for easier visualization or with the colors from the point cloud
#       line_set.paint_uniform_color([1, 0, 0])
#       # line_set.paint_uniform_color(colors[params.counter,:])
      
#       # Add the current LineSet to the main LineSet
#       params.full_line_set+=line_set
      
#       # if the counter just started add the LineSet geometry
#       if params.counter==0:
#         vis.add_geometry(params.full_line_set)
#       # else update the geometry 
#       else:
#         vis.update_geometry(params.full_line_set)
#       # update the render and counter
#       vis.update_renderer()
#       params.counter +=1
#     else:
#         # if the all point have been used reset the counter and clear the lines
#         params.counter=0
#         params.full_line_set.clear()