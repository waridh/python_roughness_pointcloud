import numpy as np
import laspy as lp

import os
import pcClass

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import simpledialog

def open_folder():
  """
  This function opens a window that lets the user select the file that is going
  to be analysed.

  Returns:
      string: The address of the file that is being analysed. Will be a las or
      laz file.
  """
  
  Tk().withdraw();
  file_name = askopenfilename(
    filetypes=[
      ("Lidar files", "*.las"),
      ("Compressed Lidar files", "*.laz"),
      ("All files", "*")
      ],
    initialdir="inputs/"
    );
  
  print("You have chosen to open the file:\n%s" % (file_name));
  return file_name;

def input_radius():
  string = simpledialog.askfloat(
    title="Roughness plane raidus search",
    prompt="What is the radius for constructing the least square plane?");
  print("The radius chosen was:", string);
  return string;

def open_folder2():
  """
  This function opens a window that lets the user select the file that is going
  to be analysed.

  Returns:
      string: The address of the file that is being analysed. Will be a las or
      laz file.
  """
  
  Tk().withdraw();
  file_name = asksaveasfilename(
    filetypes=[
    ("Lidar files", "*.las"), ("All files", "*")
    ],
    title="Save to las",
    initialdir="outputs/"
    );
  
  if file_name.endswith('.las'):
    pass;
  else:
    file_name = file_name + '.las';
  
  print("You have chosen to save to:\n%s" % (file_name));
  return file_name;

def main():
  file_name = open_folder();
  lstsqrplanradius = input_radius();
  pcCST = pcClass.PointCloudCST(file_name, lstsqrplanradius);
  pcCST.roughness2();
  save_name = open_folder2();
  pcCST.print2las(save_name);
  return;

if __name__ == "__main__":
  main();