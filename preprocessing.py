# @author: Jayant Gupta
# 6/2/2020
# Script to pre-process the wetland polygon shapefile.
# Filters to the required study area.
# Generates the necessary mask.

# Import libraries
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from rasterio import windows
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
# import gdal

import os
import shutil
import sys

from itertools import product

import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image

Image.MAX_IMAGE_PIXELS = 219494175 

import geopy.distance

import matplotlib.pyplot as plt

# Funtion to reduce the shapefile polygon region to the study region.
# The function joins the original polygon data to the county dataset (current study region).
# The join filters the data to the study region.
def preprocess():
  # Import the polygon shapefile.
  polygon_fp = "/Users/jayantgupta/Desktop/Wetland_Mapping/MN_shapefile_wetlands/MN_Wetlands_South.shp"
  polygon_data = gpd.read_file(polygon_fp)

  # Check the polygon data file.
  polygon_data.head()
  polygon_data.ATTRIBUTE.unique()
  # print(polygon_data.columns)

  # Import the county boundary file.
  county_boundary="/Users/jayantgupta/Desktop/Wetland_Mapping/shp_bdry_counties_in_minnesota/mn_county_boundaries.shp"
  county_boundary_data=gpd.read_file(county_boundary)

  county_boundary_data.head()

  # Select Hennepin and Ramsey county
  county_boundary = county_boundary_data[county_boundary_data.CTY_NAME.isin(['Hennepin', 'Ramsey'])]

  # Reproject the polygon data.
  re_polygon_data = polygon_data.to_crs(epsg=4326)

  # Reproject the county boundary data.
  re_county_boundary = county_boundary.to_crs(epsg=4326)
  re_polygon_data.head()
  re_polygon_data.size

  re_county_boundary.head()

  # Select subset of columns from the polygon dataset.
  polygons = re_polygon_data[['WETLAND_TY', 'ACRES', 'SHAPE_Leng', "SHAPE_Area", 'geometry']]
  polygons.head()

  # Select subset of columns from the county dataset.
  counties = re_county_boundary[['geometry', 'CTY_NAME']]

  # Spatial join between county and polygons.
  # This step filters the number of polygons to Hennepin and Ramsey county only.
  polygons_with_county = gpd.sjoin(polygons, counties, how="inner", op='intersects')

  polygons_with_county.head()

  print(len(polygons.index))
  print(len(polygons_with_county.index))

  polygons_with_county.to_file("/Users/jayantgupta/Desktop/Wetland_Mapping/Filtered_boundary/polygons_hc_fc_alt.shp")
  polygons_with_county.head()

# Function to project the image from one reference system to another.
def image_reprojection(in_path, out_path):
  # Reproject the raster file and save the new file.
  dst_crs = 'EPSG:4326'

  with rasterio.open(in_path) as src:
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open(out_path, 'w', **kwargs) as dst:
      for i in range(1, src.count + 1):
         reproject(
           source=rasterio.band(src, i),
           destination=rasterio.band(dst, i),
           src_transform=src.transform,
           src_crs=src.crs,
           dst_transform=transform,
           dst_crs=dst_crs,
           resampling=Resampling.nearest)  

def get_locations(polygons):
  points = polygons.geometry.centroid
  polygons['centroid'] = points

def image_masking(in_path, out_path, mask):
  with rasterio.open(in_path) as src_tr:
    out_crop, out_transform = rasterio.mask.mask(src_tr, mask.geometry, crop=False)
    out_meta = src_tr.meta
  with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(out_crop)

def binary_masking(in_path, out_path):
  print(in_path)
  print(out_path)
  masked_tif=rasterio.open(in_path)
  # Reading individual bands
  band1 = masked_tif.read(1)
  band2 = masked_tif.read(2)
  band3 = masked_tif.read(3)

  # Combining the bands
  band_combined = np.maximum(band1, band2)
  band_combined = np.maximum(band_combined, band3)

  # Filling up the masked array.
  import numpy.ma as ma

  mask = ma.array(band_combined)
  x_ind = mask.nonzero()[0]
  y_ind = mask.nonzero()[1]

  tif_mask = np.zeros((masked_tif.height, masked_tif.width), dtype=bool)
  print("mask initialized")
  print("non-zero values: ", len(x_ind))

  #non-zero values:  49705942

  for ind in range(len(x_ind)):
      tif_mask[x_ind[ind]][y_ind[ind]] = True

  size = tif_mask.shape[::-1]
#  print(size)
  databytes = np.packbits(tif_mask, axis=1)
  img = Image.frombytes(mode='1', size=size, data=databytes)
  img.save(out_path)

def mask_generation():
  polygons_with_county = gpd.read_file("/Users/jayantgupta/Desktop/Wetland_Mapping/Filtered_boundary/polygons_hc_fc_alt.shp")
  get_locations(polygons_with_county)
  print(polygons_with_county.head())
  _dir = '/Users/jayantgupta/Desktop/SV/' 
  in_path = '/Users/jayantgupta/Desktop/SV/Hennepin_North' # input_filename.
  img_dir = os.path.join(_dir, "Hennepin_North_Mask")
  if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
  os.makedirs(img_dir)
  counter = 0
  for filename in os.listdir(in_path):
    print(filename)
    if filename.endswith(".tif") is not True:
      continue
#    folder = filename.split('.')[0]
    out_path = os.path.join(img_dir, filename.split('.')[0] + "_reproject.tif")
    image_reprojection(os.path.join(in_path, filename), out_path)
  
#    poly_forested = polygons_with_county[polygons_with_county.WETLAND_TY == 'Freshwater Forested/Shrub Wetland']

    cur_in_path = os.path.join(img_dir, filename.split('.')[0] + "_reproject.tif")
    out_path = os.path.join(img_dir, filename.split('.')[0] + "_wetland.tif")
    image_masking(cur_in_path, out_path, polygons_with_county)
  
    binary_masking(out_path, os.path.join(img_dir, filename.split('.')[0] + "_mask.jpeg"))
    os.remove(cur_in_path)
    os.remove(out_path)
#    counter += 1
#    if counter == 2:
#      break

if __name__ == '__main__':
#  preprocess() # Needs to be run once in the start to create the subset dataset.
  mask_generation()
