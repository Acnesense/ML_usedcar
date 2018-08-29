import sys
from Header import *


class NN_model(object):

    def __init__(self, x_data, y_data, num_of_feature = 0, weight_of_layer = 0, learning_rate = 0.03):
#data
        self.y_data = y_data.reshape(-1,1)
        self.x_data = x_data
#parameter
        self.weight_of_car2vec = weight_of_car2vec
        self.num_of_code = num_of_code
        self.weight_of_all = weight_of_all
        self.num_of_remainfeature = num_of_remainfeature
        self.num_of_allfeature = num_of_remainfeature + 3
        self.learning_rate = learning_rate
        self.length = len(x_code)
        self.dropout = 0.7
		self.num_batch = int(len(x_data)/20)
    
    def predict(self, y_hat, y_):

        error = 100*(np.absolute(y_hat - y_))/y_
        aver_error = np.mean(error)

        return aver_error

    def shuffle(self, x_data, y_data):
        
        x_data, y_data = shuffle(x_data, y_data)
        
        return x_data, y_data       
     
    def layer(self, x_data):
      
      
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        norm_init = tf.truncated_normal_initializer (stddev=0.06) # 1/sqrt(batch_zise)?
        
      #  output1 = slim.fully_connected (x_code_ph, self.weight_of_car2vec, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init)          
        layer1 = slim.fully_connected (x_data, self.weight_of_layer, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        layer1 = slim.dropout (layer1, self.dropout, scope='dropout1')
        
       # x_embed = slim.fully_connected (output1, 3, scope='output_embed', activation_fn=None, weights_initializer=he_init)
           
       # output2 = slim.fully_connected(input2, self.weight_of_all, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        layer2 = slim.fully_connected(layer1, self.weight_of_layer, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
       
        layer2 = slim.dropout (layer2, self.dropout, scope='dropout3')
        
       # prediction = slim.fully_connected(output2, 1, scope='output_main', activation_fn=None, weights_initializer=he_init)
         
        prediction = slim.fully_connected(layer2, 1, scope='output_main', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
      
        tf.identity (prediction, name="prediction")

		return prediction
        
#        cost = tf.reduce_mean(tf.abs ((prediction-y_)))


	def train(self):


		x_data = tf.placeholder(tf.float32, [None, self.num_of_feature])
        y_data = tf.placeholder(tf.float32, [None, 1])

		prediction = layer(x_data)

        cost = tf.reduce_mean(tf.square ((prediction-y_data)))
        train = tf.train.AdamOptimizer(0.03).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.num_batch = int(len(x_train_code)/20)

        print(self.num_batch)

        
        print('=================')
        print('|Epoch\t|Error\t|')       
        print('=================')

        for i in range(500):
            for j in range(20):
                if(j<19):
                    x_data_batch = self.x_data[self.num_batch*j:self.num_batch*(j+1),:]
					y_data_batch = self.y_data[self.num_batch*j:self.num_batch*(j+1),:]
                else:
					x_data_batch = self.x_data[self.num_batch*j:,:]
					y_data_batch = self.y_data[self.num_batch*j:,:]
               	
				a, pred= sess.run((train, prediction), feed_dict={ x_data : x_data_batch , y_data : y_data_batch})
            
            x_data, y_data = self.shuffle(x_data, y_data)

            
            
            if((i+1)%100 == 0):
				y_hat = sess.run(prediction, feed_dict = {x_data : x_data_batch[0:10000]})
            	error = self.predict(y_hat, y_data[0:10000])
                print("|{:2d}\t|{:.2f}%\t|".format(i+1, error))                
               # print('epoch : ', i+1, 'error : ',error,'%')
        print('=================')
 
    


