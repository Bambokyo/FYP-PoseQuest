import numpy as np
from functions import  kmeans_clustering_nearest_neighbors,load_data_CMU,load_data, connect_neighbors, search_and_evaluate, analysis, generate_graph_html, mean_squared_error,mean_pixelwise_joint_squared_error, percentage_correct_keypoints, sliding_window


query = None
dataset = None
matrix = None
indices = None
motion = None
data_eval = None


#  step 1 
# global query, dataset, matrix, indices, motion, data_eval
# query, dataset = load_data('queryDataset.mat','wholeDataset.mat')
query, dataset = load_data_CMU('query.npy','dataset.npy')
query_shape = str(query.shape) if query is not None else None
dataset_shape = str(dataset.shape) if dataset is not None else None
# This route serves the model.html page
query_shape = str(query.shape) if query is not None else None
dataset_shape = str(dataset.shape) if dataset is not None else None
print("shapes")
print(query_shape,dataset_shape)


# step 2
K = 128
# Perform search and evaluation here, and retrieve the data
indices, data_eval , mean_mse, mean_mpjse, mean_pck , retrieval_time = search_and_evaluate(query, dataset, K)
print(data_eval)


# indices = indices.reshape(-1, 1).shape
# indices.shape

data = dataset[indices]
# data.shape
print(data.shape)
data = data[:128]
# step 3 preprocessing step 2, filter by temporal alignment
kmeans_nn = 40
new_data = kmeans_clustering_nearest_neighbors(data, kmeans_nn)
print("New data shape:", new_data.shape)


mat2 = new_data[:40,:42]
mat2.shape


# step 4    
T,graph_html, shortest_path_nodes, minimum_cost, motion = generate_graph_html(new_data)
print("shortest path and min cost: ")
print(shortest_path_nodes, minimum_cost)


# step 5
B_t , mse_b , P_t , mse_p = analysis(query, data,new_data)
# print("DTW time: ",dtw_time)


# Calculate metrics
print(motion)
print(motion.shape)

vec_array1 = motion
vec_array2 = query[:len(motion)]
mse = mean_squared_error(vec_array1, vec_array2)
mpjse = mean_pixelwise_joint_squared_error(vec_array1, vec_array2)
threshold = 10  # Set threshold for PCK
pck = percentage_correct_keypoints(vec_array1, vec_array2, threshold)


print("Mean Squared Error (MSE):", mse)
print("Mean Pixel-wise Joint Squared Error (MPJSE):", mpjse)
print("Percentage of Correct Keypoints (PCK):", pck)


# np.save(motion,"cmu_motion.npy")