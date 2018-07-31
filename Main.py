from Header import *
from Dataset import *
from Model import *

# Data Loading
Dataset = dataset(data_path = 'sonata.xlsx')

# Data removing
Dataset.feature_removal()

# Data preprocessing : OneHotIncode and Normalize
data_oh, data_nor, data_y = Dataset.feature_scaling()

NN = NN_model(x_code = data_oh, x_remain = data_nor, y_data = data_y, weight_of_car2vec = 50, num_of_code = 10, weight_of_all = 50, num_of_remainfeature = 3,  learning_rate = 0.01)
NN.train()

# Seperate train and test data
train_data, test_data = Dataset.train_test_seperate(ratio=0.8)




