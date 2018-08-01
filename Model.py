import sys
from Header import *


class NN_model(object):

    def __init__(self, x_code, x_remain, y_data, weight_of_car2vec = 0, num_of_code = 0, weight_of_all = 0, num_of_remainfeature = 0 , learning_rate = 0.03):
#data
        self.y_data = y_data.reshape(-1,1)
        self.x_code = x_code
        self.x_remain = x_remain
#parameter
        self.weight_of_car2vec = weight_of_car2vec
        self.num_of_code = num_of_code
        self.weight_of_all = weight_of_all
        self.num_of_remainfeature = num_of_remainfeature
        self.num_of_allfeature = num_of_remainfeature + 3
        self.learning_rate = learning_rate
        self.length = len(x_code)
        self.dropout = 0.7
    
    def predict(self, y_hat, y_):

        error = 100*(np.absolute(y_hat - y_))/y_
        aver_error = np.mean(error)

        return aver_error

    def shuffle(self, x_code, x_remain, y_train):
        
        x_code, x_remain, y_train = shuffle(x_code, x_remain, y_train)
        
        return x_code, x_remain, y_train       
   
    def seperate_train_test(self, x_code, x_remain, y_data, rate):
        
 #       x_code, x_remain, y_data = shuffle(x_code, x_remain, y_data)

        ratio = int(self.length*rate)
        
        x_train_code = x_code[0:ratio,:]
        x_test_code = x_code[ratio:,:]
        
        x_train_remain = x_remain[0:ratio,:]
        x_test_remain = x_remain[ratio:,:]
        
        y_train = y_data[0:ratio,:]
        y_test = y_data[ratio:,:]
        
        return x_train_code, x_test_code, x_train_remain, x_test_remain, y_train, y_test
               
 
    def train(self):
      
        x_code_ph = tf.placeholder(tf.float32, [None, self.num_of_code])
        x_remain_ph = tf.placeholder(tf.float32, [None, self.num_of_remainfeature])
        y_ = tf.placeholder(tf.float32, [None, 1])

        he_init = tf.contrib.layers.variance_scaling_initializer ()
        norm_init = tf.truncated_normal_initializer (stddev=0.06) # 1/sqrt(batch_zise)?
        
        output1 = slim.fully_connected (x_code_ph, self.weight_of_car2vec, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output1 = slim.dropout (output1, self.dropout, scope='dropout1')
        
        x_embed = slim.fully_connected (output1, 3, scope='output_embed', activation_fn=None, weights_initializer=he_init)
        input2 = tf.concat ([x_remain_ph, x_embed], 1)
        
        output2 = slim.fully_connected(input2, self.weight_of_all, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output2 = slim.dropout (output2, self.dropout, scope='dropout3')
        
        prediction = slim.fully_connected(output2, 1, scope='output_main', activation_fn=None, weights_initializer=he_init)
        tf.identity (prediction, name="prediction")
        
        cost = tf.reduce_mean(tf.abs ((prediction-y_)))
        train = tf.train.AdamOptimizer(0.03).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        x_train_code, x_test_code, x_train_remain, x_test_remain, y_train, y_test = self.seperate_train_test(self.x_code, self.x_remain, self.y_data,0.8)
        
        self.num_batch = int(len(x_train_code)/20)

        print(self.num_batch)

        for i in range(2000):
            for j in range(20):
                if(j<19):
                    x_train_code_batch = x_train_code[self.num_batch*j:self.num_batch*(j+1),:]
                    x_train_remain_batch = x_train_remain[self.num_batch*j:self.num_batch*(j+1),:]
                    y_train_batch = y_train[self.num_batch*j:self.num_batch*(j+1),:]
                    a, pred= sess.run((train, prediction), feed_dict={ x_code_ph:x_train_code_batch , x_remain_ph : x_train_remain_batch , y_: y_train_batch})
                else:
                    x_train_code_batch = x_train_code[self.num_batch*j:,:]
                    x_train_remain_batch = x_train_remain[self.num_batch*j:,:]
                    y_train_batch = y_train[self.num_batch*j:,:]
                    a, pred= sess.run((train, prediction), feed_dict={ x_code_ph:x_train_code_batch , x_remain_ph : x_train_remain_batch , y_: y_train_batch})
            

            x_train_code, x_train_remain, y_train = self.shuffle(x_train_code, x_train_remain, y_train)

            
            if((i+1)%100 == 0):
                
             
                y_hat = sess.run(prediction, feed_dict = {x_code_ph : x_train_code[0:100,:], x_remain_ph : x_train_remain[0:100,:]})
                error = self.predict(y_hat, y_train[0:100,:])

                print('epoch : ', i+1, 'error : ',error,'%')


        y_test_hat = sess.run(prediction, feed_dict = {x_code_ph : x_test_code, x_remain_ph : x_test_remain})
        test_error = self.predict(y_test_hat, y_test)

        print(' ')
        print('#############################')
        print('test error : ', test_error)
"""
plt.scatter(pred, y_test*2000)
plt.plot(pred,pred,'r')
#plt.plot(range(y_test.shape[0]), pred, label="original data")
#plt.plot(range(y_test.shape[0]), y_test*2000, label="predicted data")

plt.xlabel("Predicted price")
plt.ylabel("Actual price")
plt.show()
"""

