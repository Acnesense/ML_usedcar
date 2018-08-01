# -*- coding: utf-8 -*-

from Header import *


ohe = OneHotEncoder()

class dataset(object):

    def __init__(self, data_path = 'sonata2.xlsx'):
        self.data = np.array(pd.read_excel(data_path))

# index
        self.car_code_index = 1
        self.one_hot_index = [3,4,5]
        self.string_index = [4,5]
        
# price
        self.price_index = 7
        self.max_price = 3500
        self.min_price = 50
# accident_cost
        self.index_accident_cost = 16
        self.max_accident_cost = 25000000

# displacement
        self.index_displacement = 8
        self.max_displacement = 2000

# mile
        self.index_mile = 6
        self.max_mile = 1e+6

    def feature_removal(self):
        Preprocessor = preprocessor()
        self.data = Preprocessor.price_remove(self.data, self.price_index, self.max_price, self.min_price)
        self.data = Preprocessor.accident_cost_remove(self.data, self.index_accident_cost, self.max_accident_cost)
        self.data = Preprocessor.displacement_remove(self.data, self.index_displacement, self.max_displacement)
        self.data = Preprocessor.mile_remove(self.data, self.index_mile, self.max_mile)

    def feature_scaling(self):
        Preprocessor = preprocessor()

        Preprocessor.string_to_int(self.data, self.string_index)
         
        data_car_code = Preprocessor.OneHotEncode(data = self.data[:,self.car_code_index])
        data_remain_one_hot_data = Preprocessor.multiclass_OneHotEncode(self.data, self.one_hot_index)
        
        data_remain_nonident_data = np.c_[self.data[:,6], self.data[:,8:]]
        data_remain_nonident_data = Preprocessor.Robust_Scalar_Normalize(data_remain_nonident_data)

        data_remain = np.c_[data_remain_one_hot_data, data_remain_nonident_data]
        
        return data_car_code, data_remain, self.data[:,self.price_index]



class preprocessor:
	
    def __init__(self):
        pass

    def OneHotEncode(self, data):
        
        data = data.reshape(-1,1)
        ohe.fit(data)
        data = ohe.transform(data).toarray()

        return data

    def string_to_int(self, data, string_index):
        
        for i in range(len(string_index)):
            
            temp_data = data[:,string_index[i]]
            temp_data = temp_data.reshape(-1,1)
            temp_data = LabelEncoder().fit_transform(temp_data.ravel())
            data[:,string_index[i]] = temp_data

    def multiclass_OneHotEncode(self, data, one_hot_index):
        
        for i in range(len(one_hot_index)):
            
            temp_data = data[:,one_hot_index[i]]
            temp_data = temp_data.reshape(-1,1)
            ohe.fit(temp_data)
            temp_data = ohe.transform(temp_data).toarray()
            if(i==0):
                output = temp_data
                
            else:
                output = np.c_[output, temp_data] 
        return output

    def Robust_Scalar_Normalize(self, data):
        
        normalizer = preprocessing.RobustScaler()
        normalizer.fit(data)
        data = normalizer.transform(data)

        return data

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
