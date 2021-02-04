
######################################################################################
# @FileName             blob_detection.py
# @Brief                Helpful in identifying segmented ship pixels as one ship
#                       and calculating the ship's centroid
######################################################################################

from skimage import measure
import shapely
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

def identify_blobs(img):
    contours = measure.find_contours(img, 0.5)
    ship_polygons = []
    for contour in contours:
        y_pos = [x[0] for x in contour]
        x_pos = [x[1] for x in contour]
        ship_polygons.append(shapely.geometry.Polygon(list(zip(x_pos, y_pos))))
    return ship_polygons

def get_ship_centroid(ship_polygon):
    lat, lon = ship_polygon.centroid.y, ship_polygon.centroid.x
    return lat, lon

def show_ship_polygons(img, polygons):
    patches = []
    for polygon in polygons:
        patches.append(PolygonPatch(polygon))
    patch_collection = PatchCollection(patches)
    patch_collection.set_color('yellow')

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('./proc_img.png')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(img)
    ax.add_collection(patch_collection)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('./masked_img.png')
    plt.show()

def get_ship_coordinates(img_corner_coord, ship_centroid, lat_per_pixel, lon_per_pixel):
    lon = img_corner_coord[0] + ship_centroid[0] * lon_per_pixel
    lat = img_corner_coord[1] - ship_centroid[1] * lat_per_pixel
    return (lat, lon)