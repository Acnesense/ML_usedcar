# -*- coding: utf-8 -*-

from Header import *


ohe = OneHotEncoder()

class dataset(object):

    def __init__(self, data_path , error_path, preprocessed_error_path):
        print("Load Dataset from excel file")
        self.data = np.array(pd.read_excel(data_path))
        self.error_arr = np.array(pd.read_excel(error_path))
        self.preprocessed_error_arr = np.array(pd.read_excel(preprocessed_error_path))
        print(np.shape(self.data))
        
# index
     
        self.one_hot_index = [1,2,3,4,5,6,7,8,9,46,47,48,50]
        self.string_index = [6,8,9,46,47,48,50]
        self.price_index = 41
        self.Nan_index = [4,6,7,50,12,13,14,15,16,17,18,30,31,32]

    def feature_scaling(self):
        Preprocessor = preprocessor()

        print("Remove data for nan value")
        for i in range(len(self.Nan_index)):
            self.data = Preprocessor.delete_Nan(self.data, self.Nan_index[i])
        print(np.shape(self.data))

        print("Remove data for pricing")
        self.data = Preprocessor.price_remove(self.data, self.price_index, 50)
        print(np.shape(self.data))
        
        print("Remove data for error")
        self.data = Preprocessor.error_remove(self.data, self.error_arr)
        print(np.shape(self.data))

        print("Remove data for preprocessd error")
        self.data = Preprocessor.error_remove(self.data, self.preprocessed_error_arr)
        print(np.shape(self.data))

        print("Invert string to integer")
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
                
#        for i in range(len(remove_arr)):
#            data = np.delete(data, remove_arr[len(remove_arr)-i-1], axis=0)        
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
 
 
    def price_remove(self,data, index_price, min_price):
 
        index = np.where(data[:,index_price]<min_price)
        data = np.delete(data,index,0)
 
        return data   

    def error_remove(self, data, error_arr):

        index = np.where(error_arr >15)
        data = np.delete(data, index, 0)

        return data
