from Header import *
from Dataset import *
from Model import *

# Data Loading
Dataset = dataset(data_path = 'sonata2.xlsx')

# Data removing
Dataset.feature_removal()

# Data preprocessing : OneHotIncode and Normalize
data_car_code, data_remain, data_y = Dataset.feature_scaling()
print(data_car_code[:10])
print(data_remain[:10])
print(data_y[:10])


num_of_code = len(data_car_code[1,:])
num_of_remainfeature = len(data_remain[1,:])

NN = NN_model(x_code = data_OH, x_remain = data_nor, y_data = data_y, weight_of_car2vec = 50, num_of_code = num_of_code, weight_of_all = 50, num_of_remainfeature = num_of_remainfeature,  learning_rate = 0.01)
NN.train()


