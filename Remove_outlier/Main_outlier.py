from Header import *
from Dataset_outlier import *
from Model_outlier import *

# Data Loading
Dataset = dataset(data_path = '../Data/used_car_remove_gradecode.xlsx', error_path = '../Data/Error.xlsx', preprocessed_error_path = '../Data/Error_preprocessing.xlsx')

# Data removing
# Dataset.feature_removal()

# Data preprocessing : OneHotIncode and Normalize
x_data, y_data = Dataset.feature_scaling()

NN = NN_model(x_data,  y_data ,   weight_of_layer = 1000,  learning_rate = 0.03)
NN.train()

