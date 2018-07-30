from Header import *

ohe = OneHotEncoder()

class dataset(object):

    def __init__(self, data_path = 'sonata.xlsx'):
        self.code_index = 0
        self.data = np.array(pd.read_excel(data_path))
# price
        self.index_price = 4
        self.max_price = 3500
        self.min_price = 50
# accident_cost
        self.index_accident_cost = 3
        self.max_accident_cost = 25000000

# displacement
        self.index_displacement = 2
        self.max_displacement = 2000

# mile
        self.index_mile = 1
        self.max_mile = 1e+6

    def feature_removal(self):
        Preprocessor = preprocessor()
        self.data = Preprocessor.price_remove(self.data, self.index_price, self.max_price, self.min_price)
        self.data = Preprocessor.accident_cost_remove(self.data, self.index_accident_cost, self.max_accident_cost)
        self.data = Preprocessor.displacement_remove(self.data, self.index_displacement, self.max_displacement)
        self.data = Preprocessor.mile_remove(self.data, self.index_mile, self.max_mile)

    def feature_scaling(self):
        Preprocessor = preprocessor()

        return Preprocessor.OneHotIncode_and_Normalize(self.data, self.code_index)

    def train_test_seperate(self, ratio=0.8):
        length = int(len(self.data)*ratio)
        
        train_data = self.data[0:length,:]
        test_data = self.data[length:,:]
               
        return train_data, test_data       
class preprocessor:
	
    def __init__(self):
        pass

    def OneHotIncode_and_Normalize(self, data, index):
        data_oh = data[:,index]
        data_nor = data[:,index+1:4]
        data_y = data[:,4]

        data_oh = data_oh.reshape(-1,1)
        ohe.fit(data_oh)
        data_oh = ohe.transform(data_oh).toarray()

        normalizer = preprocessing.RobustScaler()
        normalizer.fit(data_nor)
        data_nor = normalizer.transform(data_nor)

        data = np.c_[data_oh, data_nor]
        return data_oh, data_nor, data_y

    def price_remove(self,data, index_price, max_price, min_price):

        index = np.where(data[:,index_price]<min_price)
        data = np.delete(data,index,0)

        index = np.where(data[:,index_price]>max_price)
        data = np.delete(data,index,0)
        return data

    def displacement_remove(self, data, index_displacement, max_displacement):
        
        index = np.where(data[:,index_displacement]>max_displacement)
        data = np.delete(data,index,0)

        return data

    def accident_cost_remove(self, data, index_accident_cost, max_accident_cost):

        index = np.where(data[:,index_accident_cost]>max_accident_cost)
        data = np.delete(data,index,0)

        return data

    def mile_remove(self, data, index_mile , max_mile):
     
        index = np.where(data[:,index_mile]>max_mile)
        data = np.delete(data,index,0)
        
        return data
