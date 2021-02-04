
######################################################################################
# @FileName         estimate_future_ship_pos.py
# @Brief            Estimates future coordinates of ships using AIS coordinates
######################################################################################

import numpy as np
import pandas as pd
from datetime import datetime
import math

earth_radius = 6378.1
def get_estimated_coords(row):
    ship_init_lat = row['latitude']
    ship_init_lon = row['longitude']
    speed = row['speed']
    direction = row['heading']
    ais_datetime = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
    satellite_datetime = datetime(2021, 1, 29, 7, 11, 41)
    time_diff = (satellite_datetime - ais_datetime).seconds
    distance = (((speed / 10.0) * 0.514444) * time_diff) / 1000

    lat1 = math.radians(ship_init_lat)
    lon1 = math.radians(ship_init_lon)
    bearing = math.radians(direction)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance / earth_radius) + math.cos(lat1) * math.sin(distance / earth_radius) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / earth_radius) * math.cos(lat1), math.cos(distance / earth_radius) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return row['mmsi'], (lat2, lon2)

def get_all_estimated_coords():
    ais_data = pd.read_csv('./AIS_data/ais_data_12_43.csv')
    ais_ship_pos = {}
    ais_ship_orig_pos = {}
    for _, row in ais_data.iterrows():
        mmsi, coord = get_estimated_coords(row)
        ais_ship_pos[mmsi] = coord
        mmsi, coord = row['mmsi'], (row['latitude'], row['longitude'])
        ais_ship_orig_pos[mmsi] = coord
    return ais_ship_pos, ais_ship_orig_pos