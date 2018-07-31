from Header import *
from Dataset import *
from Model import *

# Data Loading
Dataset = dataset(data_path = 'sonata.xlsx')

# Data removing
Dataset.feature_removal()

# Data preprocessing : OneHotIncode and Normalize
data_OH, data_nor, data_y = Dataset.feature_scaling()
num_of_code = len(data_OH[1,:])
num_of_remainfeature = len(data_nor[1,:])

NN = NN_model(x_code = data_OH, x_remain = data_nor, y_data = data_y, weight_of_car2vec = 50, num_of_code = num_of_code, weight_of_all = 50, num_of_remainfeature = num_of_remainfeature,  learning_rate = 0.01)
NN.train()





