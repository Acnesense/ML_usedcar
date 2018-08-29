from Header import *
from Dataset import *
#from Model_outlier import *
from Model_embedding import *
#from Model_checkpoint import *

# Data Loading
print("Load Dataset from excel file")
Dataset = dataset(data_path = 'Data/used_car_remove_gradecode.xlsx', error_path = 'Data/Error.xlsx', index_path = 'Data/car2vec_index.xlsx')
print(np.shape(Dataset.data))


print("Remove data for nan value")
Dataset.nan_remove()
print(np.shape(Dataset.data))

print("Remove data for error")
Dataset.error_remove()
print(np.shape(Dataset.data))

print("Remove data for feature scale")
Dataset.feature_remove()
print(np.shape(Dataset.data))

print("Sort by year")
Dataset.sorting_year()

print("Make string to integer")
Dataset.string_to_int()

# Data preprocessing : OneHotIncode and Normalize
"""
print("OneHotIncode and Normalize data")
x_data, y_data = Dataset.feature_scaling()


NN = NN_model(x_data,  y_data ,   weight_of_layer = 6000,  learning_rate = 0.03)
NN.train()

print("make car2vec plot data")
sonata, avante, grandeur = Dataset.car2vec_plot_data()
print(np.shape(sonata))
print(np.shape(avante))
print(np.shape(grandeur))

num_of_code = len(sonata[0,:])
num_of_remainfeature = len(sonata[0,:])

NN_embedding = NN_embedding_model(sonata, avante, grandeur, weight_of_car2vec = 1000, num_of_code = num_of_code, weight_of_all = 1000, num_of_remainfeature = num_of_remainfeature, learning_rate = 0.03)

NN_embedding.train()
"""


print("make car2vec")

x_code, x_remain, y_data = Dataset.car2vec_feature_scaling()

print(np.shape(x_code))
print(np.shape(x_remain))
print(np.shape(y_data))

num_of_code = len(x_code[0,:])
num_of_remainfeature = len(x_remain[0,:])
print(num_of_remainfeature)

NN_embedding = NN_embedding_model(x_code, x_remain, y_data, weight_of_car2vec = 1000, num_of_code = num_of_code, weight_of_all = 1000, num_of_remainfeature = num_of_remainfeature, learning_rate = 0.03)

NN_embedding.train()

