import sys
from Header import *


class NN_model(object):

    def __init__(self, x_data, y_data, weight_of_layer = 0, learning_rate = 0.03, batch_size=20):
#data
        self.y_data = y_data.reshape(-1,1)
        self.x_data = x_data
#parameter
        self.weight_of_layer = weight_of_layer
        self.learning_rate = learning_rate
        self.dropout = 0.7
        self.batch_size = batch_size
        self.num_of_feature = len(x_data[1,:])
        self.length = len(x_data)

    def predict(self, y_hat, y_):

        error = 100*(np.absolute(y_hat - y_))/y_
        aver_error = np.mean(error)

        return aver_error

    def shuffle(self, x_data, y_data):
        
        x_data, y_data = shuffle(x_data, y_data)
        
        return x_data, y_data      

   
    def seperate_data(self, x_data, y_data):
        
 #       x_code, x_remain, y_data = shuffle(x_code, x_remain, y_data)

        length1 = int(self.length*0.8)
        length2 = int(self.length*0.9)
 
        x_train = x_data[:length1,:]
        x_vali = x_data[length1:length2,:]
        x_test = x_data[length2:,:]
        
        y_train = y_data[:length1,:]
        y_vali = y_data[length1:length2,:]
        y_test = y_data[length2:,:]
 
        return x_train, x_vali, x_test, y_train, y_vali, y_test
     
    def layer(self, x_data):
      
      
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        xavier_init = tf.contrib.layers.xavier_initializer()
        norm_init = tf.truncated_normal_initializer (stddev=0.06) # 1/sqrt(batch_zise)?
        
#        layer1 = slim.fully_connected (x_code_ph, self.weight_of_car2vec, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init)          
        layer1 = slim.fully_connected (x_data, self.weight_of_layer, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=xavier_init)
        layer1 = slim.dropout (layer1, self.dropout, scope='dropout1')
        
       # x_embed = slim.fully_connected (output1, 3, scope='output_embed', activation_fn=None, weights_initializer=he_init)
           
      #  layer2 = slim.fully_connected(input2, self.weight_of_all, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        layer2 = slim.fully_connected(layer1, self.weight_of_layer, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=xavier_init)
       
        layer2 = slim.dropout (layer2, self.dropout, scope='dropout3')
        
      #  prediction = slim.fully_connected(output2, 1, scope='output_main', activation_fn=None, weights_initializer=he_init)
         
        prediction = slim.fully_connected(layer2, 1, scope='output_main', activation_fn=None, weights_initializer=xavier_init)
      
        tf.identity (prediction, name="prediction")

        return prediction


    def train(self):


        x_ = tf.placeholder(tf.float32, [None, self.num_of_feature])
        y_ = tf.placeholder(tf.float32, [None, 1])

        train_error_arr = []
        vali_error_arr = []

        prediction = self.layer(x_)

        cost = tf.reduce_mean(tf.square ((prediction-y_)))
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        x_train, x_vali, x_test, y_train, y_vali, y_test = self.seperate_data(self.x_data, self.y_data)
              
        print('=========================')
        print('|Epoch\t|Train\t|Vali\t|')       
        print('=========================')

        x_train, y_train = shuffle(x_train, y_train)
        self.num_batch = int(len(x_train)/self.batch_size)

        for i in range(1000):
            for j in range(19):
                if(j<self.batch_size-1):
                    x_train_batch = x_train[self.num_batch*j:self.num_batch*(j+1),:]
                    y_train_batch = y_train[self.num_batch*j:self.num_batch*(j+1),:]
                else:
                    x_train_batch = x_train[self.num_batch*j:,:]
                    y_train_batch = y_train[self.num_batch*j:,:]
               	
                a, pred= sess.run((train, prediction), feed_dict={ x_ : x_train_batch , y_ : y_train_batch})
          
            x_train, y_train = shuffle(x_train, y_train)

            y_train_hat = sess.run(prediction, feed_dict = {x_ : x_train[:10000,:]})
            train_error = self.predict(y_train_hat, y_train[:10000])
 
            y_vali_hat = sess.run(prediction, feed_dict = {x_ : x_vali})
            vali_error = self.predict(y_vali_hat, y_vali)
 
            train_error_arr.append(train_error)
            vali_error_arr.append(vali_error)

            if((i+1)%10 == 0):
         
                print("|{:2d}\t|{:.2f}%\t|{:.2f}%\t|".format(i+1, train_error, vali_error)) 
              
        print('=========================')

        print("finish optimization")

        y_test_hat = sess.run(prediction, feed_dict = {x_ : x_test})
        test_error = self.predict(y_test_hat, y_test)

        print("test error : {:.2f}%".format(test_error))

        plt.rcParams.update({'xtick.labelsize':'25','ytick.labelsize':'25', 'axes.labelsize' : '30'})
        plt.plot(train_error_arr,'r', linewidth = 1.5, label = 'Train_error')
        plt.plot(vali_error_arr,'b', linewidth = 1.5, label = 'Vali_error')
#        plt.hlines(11,0,500, colors = 'blue', linestyles = "--")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend(loc='upper right')
        plt.show()

