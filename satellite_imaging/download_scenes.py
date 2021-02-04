

######################################################################################
# @FileName         download_scenes.py
# @Brief            To download required scenes for a given area-of-interest (AOI)
######################################################################################

import numpy as np
import pandas as pd
from datetime import date, timedelta
import json
import subprocess
import os

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from rasterio.mask import mask as rio_mask
import shapely
import pyproj
import geopandas as gpd
import folium
import rasterio as rio

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

from utils import find_min_scenes, geojson_to_polygon, find_coverage_perc

# VARIABLES
user = 'sakshatrao'
password = 'sakshatrao'
cloud_cover_thresh = 10
scene_buffer_thresh = 0.1
NDWI_thresh = 0.3
visualize = True
convert_to_geojson = False

# Create API using username & password
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

today_date = "20210129"
tomorrow_date = "20210130"

# Query for latest Sentinel-2 scenes intersecting the ROI
footprint = geojson_to_wkt(read_geojson('geojsons/aoi.geojson'))
products = api.query(
    footprint,
    date = (today_date, tomorrow_date),
    platformname = 'Sentinel-2',
    processinglevel = 'Level-2A',
    cloudcoverpercentage = (0, cloud_cover_thresh),
)
products_gdf = api.to_geodataframe(products)
if(products_gdf.shape[0] == 0):
    print("No Data Found")
else:
    print(f"\nFound {products_gdf.shape[0]} data values")
products_gdf_sorted = products_gdf.sort_values(['size', 'beginposition'], ascending=[True, False])

# Find minimum number of scenes required
scene_polygons = list(products_gdf_sorted['geometry'].values)
rm_idx = find_min_scenes(scene_polygons, thresh = scene_buffer_thresh)
new_products_gdf_sorted = products_gdf_sorted.drop(products_gdf_sorted.iloc[rm_idx].index, axis = 0)
print(f"Final no. of scenes : {new_products_gdf_sorted.shape[0]}")

# Sort by the size of the scenes
def convert_size_to_numeric(size_str):
    size_val, size_unit = size_str.split(' ')
    if(size_unit == 'MB'):
        size_val = float(size_val) / 1000.0
    elif(size_unit == 'GB'):
        size_val = float(size_val)
    return size_val

new_products_gdf_sorted['size_GB'] = new_products_gdf_sorted['size'].apply(convert_size_to_numeric)
new_products_gdf_sorted = new_products_gdf_sorted.sort_values('size_GB')
print(f"Total Size to be downloaded: {new_products_gdf_sorted['size_GB'].sum():.2f} GB")

# Visualize the ROI, the queried scenes and the coverage
if(visualize == True):
    aoi_poly = geojson_to_polygon('./geojsons/aoi.geojson')
    scene_polygons = list(new_products_gdf_sorted['geometry'].values)
    coverage_perc = find_coverage_perc(aoi_poly, scene_polygons)

    aoi_patch = []
    scene_patches = []

    fig, ax = plt.subplots(1, 1)
    aoi_patch = [PolygonPatch(aoi_poly)]
    for idx, row in new_products_gdf_sorted.iterrows():
        scene_patches.append(PolygonPatch(row['geometry']))

    scene_patches_collection = PatchCollection(scene_patches)
    scene_patches_collection.set_color('green')
    aoi_patch_collection = PatchCollection(aoi_patch)
    aoi_patch_collection.set_color('blue')

    ax.add_collection(scene_patches_collection)
    ax.add_collection(aoi_patch_collection)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Coverage Percentage: {coverage_perc:.1f}%", fontsize = 15)
    plt.show()

# Convert all scenes to geojsons (for using with 'mapshaper')
if(convert_to_geojson == True):
    num_products = products_gdf_sorted.shape[0]
    for idx in range(num_products):
        with open(f'./products_geojsons/product{idx}.geojson', 'w') as dump_file:
            dump_file.write(gpd.GeoSeries([products_gdf_sorted.iloc[idx]['geometry']]).to_json())

# Download required scenes and extract only relevant parts of it
for product_idx, (product, row) in enumerate(new_products_gdf_sorted.iterrows()):
    print(f"Downloading Product {product_idx + 1}")
    print(f"Size: {row['size']}\n")
    
    # Check if already downloaded
    sentinel_folder = row['title'] + '.SAFE'
    if(sentinel_folder in os.listdir('./Scenes/')):
        print("\nAlready Downloaded!\n\n\n")
        continue
    
    # Download and unzip the scene
    download_info = api.download(product)
    subprocess.run(f"unzip {row['title']}.zip".split())
    subprocess.run(f"rm {row['title']}.zip".split())
    sentinel_subfolder = os.listdir(sentinel_folder + '/GRANULE/')[0]
    sentinel_subsubfolder = os.listdir(sentinel_folder + '/GRANULE/' + sentinel_subfolder + '/IMG_DATA/R10m/')[0].split('m')[0][0:-6]
    
    # Extract only 10m resolution B2, B3, B4 & B8 bands
    bands = ['B04_10m', 'B03_10m', 'B02_10m', 'B08_10m']
    subprocess.run(f"mkdir ./Scenes/{sentinel_folder}".split())
    for band in bands:
        subprocess.run(f"cp ./{sentinel_folder}/GRANULE/{sentinel_subfolder}/IMG_DATA/R10m/{sentinel_subsubfolder}{band}.jp2 ./Scenes/{sentinel_folder}/{band}.jp2".split())
    subprocess.run(f"rm -r ./{sentinel_folder}/".split())
    
    print(f"\nDownloaded Product {product_idx + 1}!\n\n\n")

new_products_gdf_sorted.to_csv('./Scenes/scenes_info.csv', index = True)
print("Completed Downloads!")