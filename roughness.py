import numpy as np
import laspy as lp

import os
import pcClass

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def open_folder():
  """
  This function opens a window that lets the user select the file that is going
  to be analysed.

  Returns:
      string: The address of the file that is being analysed. Will be a las or
      laz file.
  """
  
  Tk().withdraw();
  file_name = askopenfilename(filetypes=[("Lidar files", "*.las"),
                                          ("Compressed Lidar files", 
                                          "*.laz"), ("All files", "*")]);
  
  print("You have chosen to open the file:\n%s" % (file_name));
  return file_name;

def main():
  file_name = open_folder();
  pcCST = pcClass.PointCloudCST(file_name);
  pcCST.open3d_octree_test();
  return;

if __name__ == "__main__":
  main();