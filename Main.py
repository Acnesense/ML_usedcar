from Header import *
from Dataset import *
#from Model import *

# Data Loading
Dataset = dataset(data_path = 'sonata.xlsx')

# Data removing
Dataset.feature_removal()

# Data preprocessing : OneHotIncode and Normalize
Dataset.feature_scaling()

# Seperate train and test data
train_data, test_data = Dataset.train_test_seperate(ratio=0.8)

print(np.shape(train_data))
print(np.shape(test_data))
