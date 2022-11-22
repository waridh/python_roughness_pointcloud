import numpy as np
import laspy as lp

import os
import pcClass

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

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

def open_folder2():
  """
  This function opens a window that lets the user select the file that is going
  to be analysed.

  Returns:
      string: The address of the file that is being analysed. Will be a las or
      laz file.
  """
  
  Tk().withdraw();
  file_name = asksaveasfilename(filetypes=[
    ("Lidar files", "*.las"), ("All files", "*")
    ], title="Save to las");
  
  if file_name.endswith('.las'):
    pass;
  else:
    file_name = file_name + '.las';
  
  print("You have chosen to save to:\n%s" % (file_name));
  return file_name;

def main():
  file_name = open_folder();
  pcCST = pcClass.PointCloudCST(file_name);
  pcCST.roughness2();
  save_name = open_folder2();
  pcCST.print2las(save_name);
  return;

if __name__ == "__main__":
  main();