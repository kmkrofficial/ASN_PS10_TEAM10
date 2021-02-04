
######################################################################################
# @FileName                     get_ais_data.py
# @Brief                        To extract real-time AIS data for a particular
#                               area-of-interest (AOI)
######################################################################################

from marinetrafficapi import MarineTrafficApi
import numpy as np
import pandas as pd
from datetime import datetime

print(datetime.now())
print(datetime.utcnow())

lat = 12.2425
lon = 47.18625
gap = 9.69

api_key = 'b39c7dd60de0ab0fe5ea1c2b842287de41d082ba'
other_api_key = '7db5502fead52619cc59055edc9c2bb433d4d0c0'
api1 = '9729f36930a625231efe9e735bcabe2fd63b5195'
api2 = 'e6d415192e5ff8e9b84a4f6a10462a55efdd8342'
api3 = '1489c3b943243b30d9c4576674ade98dd24d79c7'

api = MarineTrafficApi(api_key = other_api_key)
vessels = api.fleet_vessel_positions(
    time_span = 60,
    min_latitude = lat - gap,
    max_latitude = lat + gap,
    min_longitude = lon - gap,
    max_longitude = lon + gap
)

mmsi = []
imo = []
ship_id = []
longitude = []
latitude = []
speed = []
heading = []
status = []
course = []
timestamp = []
dsrc = []
utc_seconds = []
ais_data = np.zeros((len(vessels.models), 12))
for vessel in vessels.models:
    mmsi.append(vessel.mmsi.value)
    imo.append(vessel.imo.value)
    ship_id.append(vessel.ship_id.value)
    longitude.append(vessel.longitude.value)
    latitude.append(vessel.latitude.value)
    speed.append(vessel.speed.value)
    heading.append(vessel.heading.value)
    status.append(vessel.status.value)
    course.append(vessel.course.value)
    timestamp.append(vessel.timestamp.value)
    dsrc.append(vessel.dsrc.value)
    utc_seconds.append(vessel.utc_seconds.value)

ais_dict = {
    'mmsi': mmsi,
    'imo': imo,
    'ship_id': ship_id,
    'longitude': longitude,
    'latitude': latitude,
    'speed': speed,
    'heading': heading,
    'status': status,
    'course': course,
    'timestamp': timestamp,
    'dsrc': dsrc,
    'utc_seconds': utc_seconds
}
ais_data = pd.DataFrame.from_dict(ais_dict)
ais_data.to_csv('./ais_data_12_43.csv', index = False)