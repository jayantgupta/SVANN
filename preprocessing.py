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
from tqdm import tqdm

import os
import shutil
import sys

from itertools import product

import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import json

Image.MAX_IMAGE_PIXELS = 219494175 

import geopy.distance

import matplotlib.pyplot as plt

def generate_imagery_tiles(tile_width, tile_height, 
                        in_dir_root=json.load(open('config.json'))['filepaths']['default_reprojections_dir'], 
                        in_folder=''):
    current_dir = os.path.join(in_dir_root, in_folder)
    for filename in tqdm(os.listdir(current_dir)):
        in_file = os.path.join(current_dir, filename)
        if os.path.isdir(in_file) and filename != 'tiles':
            # Recursively call generate_imagery_tiles for processing folders
            generate_imagery_tiles(tile_width, tile_height, in_dir_root, in_folder=os.path.join(in_folder, filename))  
        if in_file.endswith(".tif") is not True:
            continue
        out_dir = os.path.join(in_dir_root, 'tiles')
        out_path = os.path.join(out_dir, os.path.join(in_folder, filename.split('.')[0]))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        print(in_file)
        with rasterio.open(in_file) as inds:
            meta = inds.meta.copy()
            for window, transform in get_tiles(inds, tile_width, tile_height):
                if window.width != tile_width or window.height != tile_height:
                  continue
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                out_file = os.path.join(out_path, 'tile_{}-{}.tif'.format(int(window.col_off), int(window.row_off)))
                #jpeg_out_file = os.path.join(out_path, 'tile_{}-{}.jpeg'.format(int(window.col_off), int(window.row_off)))
                if os.path.exists(out_file):
                  continue
                with rasterio.open(out_file, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
                #convert_to_JPEG(out_file, jpeg_outpath)
                #os.remove(out_file)  

def generate_mask_tiles(tile_width, tile_height, 
                        in_dir_root=json.load(open('config.json'))['filepaths']['default_masks_dir'], 
                        in_folder=''):
    current_dir = os.path.join(in_dir_root, in_folder)
    for filename in tqdm(os.listdir(current_dir)):
        in_file = os.path.join(current_dir, filename)
        if os.path.isdir(in_file) and filename != 'tiles':
            # Recursively call generate_imagery_tiles for processing folders
            generate_mask_tiles(tile_width, tile_height, in_dir_root, in_folder=os.path.join(in_folder, filename))  
        if filename.endswith(".jpeg") is not True:
            continue
        out_dir = os.path.join(in_dir_root, 'tiles')
        out_path = os.path.join(out_dir, os.path.join(in_folder, filename.split('.')[0]))
        if not os.path.exists(out_path):
          os.makedirs(out_path)

        print(in_file)
        img = Image.open(in_file)
        img_width, img_height = img.size
        
        for col_i in range(0, img_width-tile_width, tile_width):
            for row_i in range(0, img_height-tile_height, tile_height):
                crop = img.crop((col_i, row_i, col_i + tile_width, row_i + tile_height))
                if crop.width != tile_width or crop.height != tile_height:
                  continue
                out_file = os.path.join(out_path, 'tile_{}-{}.jpeg'.format(int(col_i), int(row_i)))
                if os.path.exists(out_file):
                  img = Image.open(out_file)
                  if img.width != tile_width or img.height != tile_height:
                    os.remove(out_file)
                  continue
                crop.save(out_file)

def get_tiles(ds, width=256, height=256):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
	
def convert_to_JPEG(tif_filename, jpeg_filename):
    with rasterio.open(tif_filename) as infile:
        profile=infile.profile
        profile['driver']='JPEG'
        raster=infile.read()
        with rasterio.open(jpeg_filename, 'w', **profile) as dst:
            dst.write(raster)

def partition_inputs(tile_width, tile_height):
    generate_imagery_tiles(tile_width, tile_height)
    generate_mask_tiles(tile_width, tile_height)  

# Funtion to reduce the shapefile polygon region to the study region.
# The function joins the original polygon data to the county dataset (current study region).
# The join filters the data to the study region.
def preprocess(polygon_path=json.load(open('config.json'))['filepaths']['input_polygon'], 
               boundary_path=json.load(open('config.json'))['filepaths']['input_county_boundary'], 
               out_path=json.load(open('config.json'))['filepaths']['default_filtered_polygon']):
  polygon_data = gpd.read_file(polygon_path)
  
  # Check the polygon data file.
  polygon_data.head()
  polygon_data.ATTRIBUTE.unique()
  # print(polygon_data.columns)
  county_boundary_data=gpd.read_file(boundary_path)
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

  polygons_with_county.to_file(out_path + "/polygons_hc_fc_alt.shp")
  polygons_with_county.head()

# Function to project the image from one reference system to another.
def image_reprojection(in_path, out_path):
  # Reproject the raster file and save the new file.
  dst_crs = 'EPSG:4326'

  out_dir = os.path.dirname(out_path)
  if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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
  if not os.path.exists(out_path.split('.')[0]):
    os.makedirs(out_path.split('.')[0])

  with rasterio.open(in_path) as src_tr:
    out_crop, out_transform = rasterio.mask.mask(src_tr, mask.geometry, crop=False)
    out_meta = src_tr.meta
  with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(out_crop)

def binary_masking(in_path, out_path):
  print(in_path)
  print(out_path)
  with rasterio.open(in_path)as masked_tif:
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

def mask_generation(in_dir_root=json.load(open('config.json'))['filepaths']['input_imagery_dir'],
                    in_folder='',
                    polygon_path=json.load(open('config.json'))['filepaths']['default_filtered_polygon'],
                    reprojection_dir=json.load(open('config.json'))['filepaths']['default_reprojections_dir'],
                    out_dir=json.load(open('config.json'))['filepaths']['default_masks_dir']):
  polygons_with_county = gpd.read_file(polygon_path)
  get_locations(polygons_with_county)
  
  current_dir = os.path.join(in_dir_root, in_folder)
  for filename in tqdm(os.listdir(current_dir)):
    in_file = os.path.join(current_dir, filename)
    if os.path.isdir(in_file) and filename != 'tiles':
      # Recursively call mask_generation for processing folders
      mask_generation(in_dir_root=in_dir_root, in_folder=os.path.join(in_folder, filename), polygon_path=polygon_path, out_dir=out_dir)
    if in_file.endswith(".tif") is not True:
      continue
    out_path = os.path.join(out_dir, os.path.join(in_folder, filename.split('.')[0]))
    #if os.path.exists(out_path + "_mask.jpeg"):
    #  continue
    reprojection_path = os.path.join(reprojection_dir, in_folder)
    reprojection_file = os.path.join(reprojection_path, filename)
    image_reprojection(in_file, reprojection_file)
  
    wetland_path = out_path + "_wetland.tif"
    image_masking(reprojection_file, wetland_path, polygons_with_county)
  
    binary_masking(wetland_path, out_path + ".jpeg")
    os.remove(wetland_path)

if __name__ == '__main__':
  #preprocess()      # Needs to be run once in the start to create the subset dataset.
  #mask_generation()
  partition_inputs(1024, 1024)
