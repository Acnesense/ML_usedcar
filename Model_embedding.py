from Header import *


class NN_embedding_model(object):

    def __init__(self, x_code, x_remain, y_data, weight_of_car2vec = 0, num_of_code = 0, weight_of_all = 0, num_of_remainfeature = 0 , learning_rate = 0.03):
#data
        self.y_data = y_data.reshape(-1,1)
        self.x_code = x_code
        self.x_remain = x_remain

        self.sonata = np.array([])
        self.avante = np.array([])
        self.grandeur = np.array([])
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
   
    def seperate_train_test(self, x_code, x_remain, y_data):
        
 #       x_code, x_remain, y_data = shuffle(x_code, x_remain, y_data)

        length1 = int(self.length*0.8)
        length2 = int(self.length*0.9)


        x_train_code = x_code[:length1,:]
        x_vali_code = x_code[length1:length2,:]
        x_test_code = x_code[length2:,:]
 
        x_train_remain = x_remain[:length1,:]
        x_vali_remain = x_remain[length1:length2,:]
        x_test_remain = x_remain[length2:,:]
        
        y_train = y_data[:length1,:]
        y_vali = y_data[length1:length2,:]
        y_test = y_data[length2:,:]
        
        return x_train_code, x_vali_code, x_test_code,  x_train_remain,x_vali_remain, x_test_remain, y_train,y_vali, y_test
               
 
    def train(self):
      
        x_code_ph = tf.placeholder(tf.float32, [None, self.num_of_code])
        x_remain_ph = tf.placeholder(tf.float32, [None, self.num_of_remainfeature])
        y_ = tf.placeholder(tf.float32, [None, 1])
        error_arr = []
        arr_12 = []

        he_init = tf.contrib.layers.variance_scaling_initializer ()
        norm_init = tf.truncated_normal_initializer (stddev=0.06) # 1/sqrt(batch_zise)?
        
      #  output1 = slim.fully_connected (x_code_ph, self.weight_of_car2vec, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init)          
        output1 = slim.fully_connected (x_code_ph, self.weight_of_car2vec, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        output1 = slim.dropout (output1, self.dropout, scope='dropout1')
        
       # x_embed = slim.fully_connected (output1, 3, scope='output_embed', activation_fn=None, weights_initializer=he_init)
         
        x_embed = slim.fully_connected (output1, 3, scope='output_embed', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        input2 = tf.concat ([x_remain_ph, x_embed], 1)
        
       # output2 = slim.fully_connected(input2, self.weight_of_all, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output2 = slim.fully_connected(input2, self.weight_of_all, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
       

        output2 = slim.dropout (output2, self.dropout, scope='dropout3')
        
       # prediction = slim.fully_connected(output2, 1, scope='output_main', activation_fn=None, weights_initializer=he_init)
         
        prediction = slim.fully_connected(output2, 1, scope='output_main', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
      
        tf.identity (prediction, name="prediction")
        
#        cost = tf.reduce_mean(tf.abs ((prediction-y_)))
        cost = tf.reduce_mean(tf.square ((prediction-y_)))
        train = tf.train.AdamOptimizer(0.03).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        saver.restore(sess, '../model1')

        i = 0
        for idx in self.y_data:

            if(i % 1000 == 0):
                print(i)

            if(idx == 1):
                if(len(self.sonata) == 0):
                    self.sonata = self.x_code[i,:]
                else:
                    self.sonata = np.vstack([self.sonata, self.x_code[i,:]])

            elif(idx == 2):
                
                if(len(self.avante) == 0):
                    self.avante = self.x_code[i,:]
                else:
                    self.avante = np.vstack([self.avante, self.x_code[i,:]])

            elif(idx == 3):
                if(len(self.grandeur) == 0):
                    self.grandeur = self.x_code[i,:]
                else:
                    self.grandeur = np.vstack([self.grandeur, self.x_code[i,:]])

            i=i+1
            if(i==10000):
                break

        print(np.shape(self.sonata))
        print(np.shape(self.avante))
        print(np.shape(self.grandeur))

        sonata_car2vec = sess.run(x_embed, feed_dict = {x_code_ph : self.sonata})

        avante_car2vec = sess.run(x_embed, feed_dict = {x_code_ph : self.avante})

        grandeur_car2vec = sess.run(x_embed, feed_dict = {x_code_ph : self.grandeur})


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for data in sonata_car2vec:
            ax.scatter(data[0], data[1], data[2], c='r', marker='o')

        for data in avante_car2vec:
            ax.scatter(data[0], data[1], data[2], c='b', marker='v')

        for data in grandeur_car2vec:
            ax.scatter(data[0], data[1], data[2], c='black', marker='*')

        plt.show()


"""
        x_train_code,x_vali_code, x_test_code, x_train_remain, x_vali_remain, x_test_remain, y_train, y_vali, y_test = self.seperate_train_test(self.x_code, self.x_remain, self.y_data)
        print(len(self.x_code))      
        self.num_batch = int(len(x_train_code)/20)

        print(self.num_batch)

        
        print('=================')
        print('|Epoch\t|Error\t|')       
        print('=================')

        for i in range(400):
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

            
              
            y_hat = sess.run(prediction, feed_dict = {x_code_ph : x_vali_code, x_remain_ph : x_vali_remain})
            error = self.predict(y_hat, y_vali)
#            error_arr.append(error)
#            arr_12.append(12)
            if((i+1)%10 == 0):
                print("|{:2d}\t|{:.2f}%\t|".format(i+1, error))                
               # print('epoch : ', i+1, 'error : ',error,'%')
        saver = tf.train.Saver()
        saver.save(sess, '../model2.ckpt')
        
        print('=================')
 
        y_test_hat = sess.run(prediction, feed_dict = {x_code_ph : x_test_code, x_remain_ph : x_test_remain})
        test_error = self.predict(y_test_hat, y_test)

        print(' ')
        print('### optimization is finished ###')
        print('test error : ', test_error)

        plt.rcParams.update({'xtick.labelsize':'25','ytick.labelsize':'25', 'axes.labelsize' : '30'})
        plt.plot(error_arr,'r', linewidth = 1.5)
      #  plt.plot(arr_12,'b', linestyle = "--")
#plt.plot(range(y_test.shape[0]), pred, label="original data")
#plt.plot(range(y_test.shape[0]), y_test*2000, label="predicted data")
        plt.hlines(11,0,500, colors = 'blue', linestyles = "--")
        plt.xlabel("Iteration")
        plt.ylabel("Valid Error")
        plt.show()
"""

