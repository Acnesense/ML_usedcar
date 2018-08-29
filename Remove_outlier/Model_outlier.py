import sys
from Header import *


class NN_model(object):

    def __init__(self, x_data, y_data, weight_of_layer = 0, learning_rate = 0.03):
#data
        self.y_data = y_data.reshape(-1,1)
        self.x_data = x_data
#parameter
        self.weight_of_layer = weight_of_layer
        self.learning_rate = learning_rate
        self.dropout = 0.7
        self.batch_size = 20
        self.num_batch = int(len(x_data)/self.batch_size)
        self.num_of_feature = len(x_data[1,:])

    def predict(self, y_hat, y_):

        error = 100*(np.absolute(y_hat - y_))/y_
        aver_error = np.mean(error)

        return aver_error

    def shuffle(self, x_data, y_data):
        
        x_data, y_data = shuffle(x_data, y_data)
        
        return x_data, y_data       
     
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
        
#        cost = tf.reduce_mean(tf.abs ((prediction-y_)))


    def train(self):


        x_ = tf.placeholder(tf.float32, [None, self.num_of_feature])
        y_ = tf.placeholder(tf.float32, [None, 1])

        prediction = self.layer(x_)

        cost = tf.reduce_mean(tf.square ((prediction-y_)))
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        print(self.num_batch)

              
        print('=================')
        print('|Epoch\t|Error\t|')       
        print('=================')
        x_data, y_data = shuffle(self.x_data, self.y_data)


        for i in range(400):
            for j in range(self.batch_size):
                if(j<self.batch_size-1):
                    x_data_batch = x_data[self.num_batch*j:self.num_batch*(j+1),:]
                    y_data_batch = y_data[self.num_batch*j:self.num_batch*(j+1),:]
                else:
                    x_data_batch = x_data[self.num_batch*j:,:]
                    y_data_batch = y_data[self.num_batch*j:,:]
               	
                a, pred= sess.run((train, prediction), feed_dict={ x_ : x_data_batch , y_ : y_data_batch})
          
            x_data, y_data = shuffle(self.x_data, self.y_data)

            if((i+1)%10 == 0):
                y_hat = sess.run(prediction, feed_dict = {x_ : x_data[0:10000]})
                error = self.predict(y_hat, y_data[0:10000])
                print("|{:2d}\t|{:.2f}%\t|".format(i+1, error))                
               # print('epoch : ', i+1, 'error : ',error,'%')
        print('=================')

        y_hat = sess.run(prediction, feed_dict = {x_ : self.x_data})
        error = 100*(np.absolute(y_hat - self.y_data))/self.y_data

        print(error)
"""
        df = pd.DataFrame(error)
        writer = pd.ExcelWriter('../Data/Error_preprocessing.xlsx', engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
"""        
                
 
