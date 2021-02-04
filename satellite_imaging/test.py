
######################################################################################
# @FileName         test.py
# @Brief            Tests the ship detection model and plots coordinates of ships
#                   along with other AIS-verified ships
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from apply_ship_detection import Model_Applicator
from blob_detection import *
from estimate_future_ship_pos import get_all_estimated_coords, earth_radius
from tqdm.notebook import tqdm

# Scene 4: (2, 33)
# Scene 5: (41, 46), (43, 40), (47, 33)
# Scene 6: (29, 17), x(22, 22)
# Scene 8: (41, 1)

folder4 = 'S2A_MSIL2A_20210129T071141_N0214_R106_T38PQU_20210129T102352.SAFE'
folder5 = 'S2A_MSIL2A_20210129T071141_N0214_R106_T38PQV_20210129T102352.SAFE'
folder6 = 'S2A_MSIL2A_20210129T071141_N0214_R106_T38PRT_20210129T102352.SAFE'
folder8 = 'S2A_MSIL2A_20210129T071141_N0214_R106_T38PRV_20210129T102352.SAFE'
sentinel_folder = folder5

model_appl = Model_Applicator(sentinel_folder)
model_appl.extract_satellite_img()

coords, full_img = model_appl.get_img_coordinates()

batch_size = 8
ship_area_thresh = 20

output_mask = np.zeros((full_img.shape[0], full_img.shape[1]), dtype = np.uint8)
detected_ships = {}

lon_per_pixel = (coords[1][0] - coords[0][0]) / full_img.shape[1]
lat_per_pixel = (coords[0][1] - coords[2][1]) / full_img.shape[0]

batch = np.zeros((batch_size, 224, 224, 3), dtype = np.float32)
imgs = np.zeros((batch_size, 224, 224, 3), dtype = np.uint8)
current_batch_idx = []
batch_cnt = 0
for j in tqdm(range(full_img.shape[0] // 224), total = full_img.shape[0] // 224):
    for i in range(full_img.shape[1] // 224):
        sub_img = full_img[224*j:224*j+224, 224*i:224*i+224, :]
        sub_img = model_appl.process_img(sub_img)
        imgs[batch_cnt, :, :, :] = sub_img
        batch[batch_cnt, :, :, :] = sub_img
        current_batch_idx.append((j, i))
        batch_cnt += 1
        if(batch_cnt == batch_size):
            processed_batch = model_appl.batch_preprocessor(batch)
            outputs = model_appl.apply_model(processed_batch)
            for output_cnt in range(batch_size):
                sub_output = outputs[output_cnt, :, :]
                output_img = imgs[output_cnt, :, :, :]
                batch_j, batch_i = current_batch_idx[output_cnt]
                batch_i *= 224; batch_j *= 224

                output_mask[batch_j:batch_j+224, batch_i:batch_i+224] = sub_output
                
                ship_polygons = identify_blobs(sub_output.astype(np.uint8))
                ship_centroids = [get_ship_centroid(polygon) for polygon in ship_polygons]
                
                sub_img_corner_coord = (coords[0][0] + batch_i * lon_per_pixel, coords[0][1] - batch_j * lat_per_pixel)

                if(len(ship_centroids) != 0):
                    #show_ship_polygons(output_img, ship_polygons)
                    for ship_idx in range(len(ship_centroids)):
                        if(ship_polygons[ship_idx].area > ship_area_thresh):
                            ship_coord = get_ship_coordinates(sub_img_corner_coord, ship_centroids[ship_idx], lat_per_pixel = lat_per_pixel, lon_per_pixel = lon_per_pixel)
                            detected_ships[ship_coord] = ship_polygons[ship_idx]
                
            batch = np.zeros((batch_size, 224, 224, 3), dtype = np.float32)
            batch_cnt = 0
            current_batch_idx = []

processed_batch = model_appl.batch_preprocessor(batch)
outputs = model_appl.apply_model(processed_batch)
for output_cnt in range(len(current_batch_idx)):
    sub_output = outputs[output_cnt, :, :]
    output_img = imgs[output_cnt, :, :, :]
    batch_j, batch_i = current_batch_idx[output_cnt]
    batch_i *= 224; batch_j *= 224

    output_mask[batch_j:batch_j+224, batch_i:batch_i+224] = sub_output
    
    ship_polygons = identify_blobs(sub_output.astype(np.uint8))
    ship_centroids = [get_ship_centroid(polygon) for polygon in ship_polygons]
    
    sub_img_corner_coord = (coords[0][0] + batch_i * lon_per_pixel, coords[0][1] - batch_j * lat_per_pixel)

    if(len(ship_centroids) != 0):
        show_ship_polygons(output_img, ship_polygons)
        for ship_idx in range(len(ship_centroids)):
            ship_coord = get_ship_coordinates(sub_img_corner_coord, ship_centroids[ship_idx], lat_per_pixel = lat_per_pixel, lon_per_pixel = lon_per_pixel)
            detected_ships[ship_coord] = ship_polygons[ship_idx]

np.save("./ship_mask.npy", output_mask)

x_idx = 43 * 224
y_idx = 40 * 224
fig, ax = plt.subplots(1, 1, figsize = (20, 20))
ship_img = full_img[x_idx: x_idx + 1 * 224, y_idx: y_idx + 1 * 224, :]
ship_img = model_appl.process_img(ship_img)
ax.imshow(ship_img)
plt.show()
ship_output_mask = output_mask[x_idx: x_idx + 1 * 224, y_idx: y_idx + 1 * 224]
fig, ax = plt.subplots(1, 1, figsize = (20, 20))
ax.imshow(ship_output_mask, cmap = 'gray')
plt.show()

batch = np.asarray([ship_img])
processed_batch = model_appl.batch_preprocessor(batch)
outputs = model_appl.apply_model(processed_batch)

sub_output = outputs[0, :, :]
ship_polygons = identify_blobs(sub_output.astype(np.uint8))
ship_centroids = [get_ship_centroid(polygon) for polygon in ship_polygons]
#show_ship_polygons(ship_img, ship_polygons)

lon_per_pixel = (coords[1][0] - coords[0][0]) / full_img.shape[1]
lat_per_pixel = (coords[0][1] - coords[2][1]) / full_img.shape[0]
sub_img_corner_coord = (coords[0][0] + y_idx * lon_per_pixel, coords[0][1] - x_idx * lat_per_pixel)

ship_coords = []
if(len(ship_centroids) != 0):
    for ship_idx in range(len(ship_centroids)):
        ship_coords.append(get_ship_coordinates(sub_img_corner_coord, ship_centroids[ship_idx], lat_per_pixel = lat_per_pixel, lon_per_pixel = lon_per_pixel))

print(f"Ship Coordinates: {ship_coords}")

import math
import pandas as pd
def get_coords_dist(coord1, coord2):
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = earth_radius * c
    return distance

ais_ship_coords, ais_ship_orig_pos = get_all_estimated_coords()
ais_ship_coords = list(ais_ship_coords.items())
ais_ship_orig_pos = list(ais_ship_orig_pos.items())
distance_from_ais_coords = [(mmsi_coord[0], get_coords_dist(mmsi_coord[1], ship_coords[0])) for mmsi_coord in ais_ship_coords]
sorted_distances = list(sorted(distance_from_ais_coords, key = lambda x: x[1]))
sorted_distances_df = pd.DataFrame(sorted_distances, columns = ['MMSI', 'Distance'])
sorted_distances_df.head()

plt.scatter([x[1][1] for x in ais_ship_coords], [x[1][0] for x in ais_ship_coords], label = 'Coordinates from AIS data')
plt.scatter([ship_coords[0][1]], [ship_coords[0][0]], color = 'red', label = 'Observed Ship Coordinates')
plt.grid('on')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc = 'best')
plt.savefig('./coords.png')
plt.show()