# -*- coding: utf-8 -*-

from Header import *


ohe = OneHotEncoder()

class dataset(object):

    def __init__(self, data_path , error_path, preprocessed_error_path):
        self.data = np.array(pd.read_excel(data_path))
        self.error_arr = np.array(pd.read_excel(error_path))
        self.preprocessed_error_arr = np.array(pd.read_excel(preprocessed_error_path))

# index
        self.one_hot_index = [1,2,3,4,5,6,7,8,9,46,47,48,50]
        self.string_index = [6,8,9,46,47,48,50]
        self.price_index = 41
        self.Nan_index = [4,6,7,50,12,13,14,15,16,17,18,30,31,32]
        
# price
        self.index_price = 41
        self.max_price = 30000
        self.min_price = 50
# accident_cost
        self.index_accident_cost = 34
        self.max_accident_cost = 1e+8

# displacement
        self.index_displacement = 11
        self.max_displacement = 10000

# mile
        self.index_mile = 10
        self.max_mile = 1e+6

# year
        self.index_year = 7
        self.min_year = 2000
        self.max_year = 2018

# sales completion days
        self.index_sales = 45

    def nan_remove(self):

        Preprocessor = preprocessor()
		
        for i in range(len(self.Nan_index)):
            self.data = Preprocessor.delete_Nan(self.data, self.Nan_index[i])

    def error_remove(self):
        Preprocessor = preprocessor()

        self.data = Preprocessor.minprice_remove(self.data, self.price_index, 50)
        self.data = Preprocessor.error_remove(self.data, self.error_arr)
    #    self.data = Preprocessor.error_remove(self.data, self.preprocessed_error_arr)

    def feature_remove(self):
        Preprocessor = preprocessor()

        self.data = Preprocessor.price_remove(self.data, self.index_price, self.max_price, self.min_price)
        self.data = Preprocessor.year_remove(self.data, self.index_year, self.max_year, self.min_year)
        self.data = Preprocessor.accident_cost_remove(self.data, self.index_accident_cost, self.max_accident_cost)
        self.data = Preprocessor.displacement_remove(self.data, self.index_displacement, self.max_displacement)
        self.data = Preprocessor.mile_remove(self.data, self.index_mile, self.max_mile)

    def sorting_year(self):
        
        Preprocessor = preprocessor()

        self.data = Preprocessor.sorting(self.data, self.index_sales)
        print(self.data[100:300,45])


    def feature_scaling(self):
        Preprocessor = preprocessor()

        Preprocessor.string_to_int(self.data, self.string_index)


        print("Encoding data to One Hot class")
        one_hot_data = Preprocessor.multiclass_OneHotEncode(self.data, self.one_hot_index)
        
        print("Normalize remain data")
       # remain_data = np.c_[self.data[:,10:33], self.data[:,34]]
        remain_data = self.data[:,10:35]
        remain_data = Preprocessor.Robust_Scalar_Normalize(remain_data)

        x_data = np.c_[one_hot_data, remain_data]
        y_data = self.data[:,self.price_index]
        print(np.shape(x_data))

        return x_data, y_data

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


    def delete_Nan(self, data, feature_index):
        remove_arr = []
        for i in range(len(data[:,feature_index])):

            if(pd.isnull(np.array(data[i,feature_index]))):
                remove_arr.append(i)
        data = np.delete(data, remove_arr, 0)         
        
        return data    

    def multiclass_OneHotEncode(self, data, one_hot_index):
        
               
        for i in range(len(one_hot_index)):
            
            temp_data = data[:,one_hot_index[i]]
            temp_data = temp_data.reshape(-1,1)
            temp_data = temp_data.astype(int)
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

    def year_remove(self,data, index_year, max_year, min_year):

        index = np.where(data[:,index_year]<min_year)
        data = np.delete(data,index,0)

        index = np.where(data[:,index_year]>max_year)
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

    def error_remove(self, data, error_arr):
        index = np.where(error_arr > 15)
        data = np.delete(data, index, 0)

        return data

    def minprice_remove(self,data, index_price, min_price):
 
        index = np.where(data[:,index_price]<min_price)
        data = np.delete(data,index,0)
 
        return data   

    def sorting(self, data, index):

        data = data[data[:,index].argsort()]

        return data


