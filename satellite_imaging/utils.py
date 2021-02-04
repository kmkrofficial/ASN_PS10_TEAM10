
######################################################################################
# @FileName             utils.py
# @Brief                Helpful functions
######################################################################################

from shapely.geometry import Polygon, MultiPolygon
from sentinelsat import read_geojson
import numpy as np
import json

def geojson_to_polygon(geojson_path):
    def find_coord(geojson):
        for key in geojson:
            if(key == 'coordinates'):
                coords = geojson[key]
                while(1):
                    if(isinstance(coords, list)):
                        if(len(coords) == 1):
                            coords = coords[0]
                        else:
                            break
                    else:
                        break
                return coords
            if(isinstance(geojson[key], dict)):
                return_val = find_coord(geojson[key])
                if(return_val != None):
                    return return_val
            elif(isinstance(geojson[key], list)):
                if(isinstance(geojson[key][0], dict)):
                    return_val = find_coord(geojson[key][0])
                    if(return_val != None):
                        return return_val
                else:
                    pass
            else:
                pass
        return None
    geojson = read_geojson(geojson_path)
    coords = find_coord(geojson)
    assert(coords != None)
    return MultiPolygon([Polygon(coords)])

def find_min_scenes(scene_polygons, thresh = 0.1):
    roi_poly = geojson_to_polygon("./geojsons/south_china_sea.geojson")
    intersecs = []
    for poly in scene_polygons:
        #poly = geojson_to_polygon(f"./products_geojsons/product{idx + 1}.geojson")
        intersecs.append(roi_poly.intersection(poly))
    
    rm_idx = []
    for idx1, intersec1 in enumerate(intersecs):
        for idx2, intersec2 in enumerate(intersecs):
            if(idx1 < idx2):
                if(intersec1.buffer(thresh).contains(intersec2)):
                    rm_idx.append(idx2)
                elif(intersec2.buffer(thresh).contains(intersec1)):
                    rm_idx.append(idx1)
    return list(set(rm_idx))

def find_coverage_perc(roi_poly, polys):
    union_poly = polys[0]
    for idx in range(1, len(polys)):
        union_poly = union_poly.union(polys[idx])
    union_poly = union_poly.intersection(roi_poly)
    return (union_poly.area / roi_poly.area * 100.0)