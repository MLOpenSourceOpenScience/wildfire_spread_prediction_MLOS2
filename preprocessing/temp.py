import numpy as np
from sklearn.cluster import DBSCAN

# Earth's radius in kilometers
R = 6371.0

# Function to convert degrees to radians
def deg2rad(degrees):
    return degrees * np.pi / 180

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Clustering fire points
fire_lat_lon = fire_points[:, [2, 3]]  # Adjust these indices based on your data
dbscan = DBSCAN(eps=0.05, min_samples=10, metric=lambda u, v: haversine(u[0], u[1], v[0], v[1]))
clusters = dbscan.fit_predict(fire_lat_lon)

# Calculate centroids of clusters
centroids = []
for cluster_id in np.unique(clusters):
    if cluster_id != -1:
        index = clusters == cluster_id
        centroid_lat = np.mean(fire_lat_lon[index, 0])
        centroid_lon = np.mean(fire_lat_lon[index, 1])
        centroids.append((centroid_lat, centroid_lon))

# Create a mask for points within 20 km from any centroid
proximity_mask = np.zeros(len(combined_data), dtype=int)

for centroid in centroids:
    # Each centroid processed
    centroid_lat, centroid_lon = centroid
    for i, point in enumerate(combined_data):
        point_lat, point_lon = point[2], point[3]  # Adjust indices if necessary
        if haversine(centroid_lat, centroid_lon, point_lat, point_lon) <= 20:
            proximity_mask[i] = 1

# Add the mask as a new column to combined_data
combined_data_with_mask = np.hstack((combined_data, proximity_mask[:, None]))

print(combined_data_with_mask.shape)  # This will show the original number of rows and one additional column
