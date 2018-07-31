from Header import *


class NN_model(object):

    def __init__(self, x_code, x_remain, y_train, weight_of_car2vec = 0, num_of_code = 0, weight_of_all = 0, num_of_remainfeature = 0 , learning_rate = 0.03):
#data
        self.y_train = y_train.reshape(-1,1)
        self.x_code = x_code
        self.x_remain = x_remain
#parameter
        self.weight_of_car2vec = weight_of_car2vec
        self.num_of_code = num_of_code
        self.weight_of_all = weight_of_all
        self.num_of_remainfeature = num_of_remainfeature
        self.num_of_allfeature = num_of_remainfeature + 3
        self.learning_rate = learning_rate
        self.num_batch = 450
    
    def predict(self, y_hat, y_):

        error = 100*(np.absolute(y_hat - y_))#/y_
        aver_error = np.mean(error)

        return aver_error


    def train(self):
      
        x_code_ph = tf.placeholder(tf.float32, [None, self.num_of_code])
        x_remain_ph = tf.placeholder(tf.float32, [None, self.num_of_remainfeature])
        y_ = tf.placeholder(tf.float32, [None, 1])

        w_1 = tf.Variable(tf.truncated_normal([self.num_of_code, self.weight_of_car2vec]))
        b_1 = tf.Variable(tf.zeros([self.weight_of_car2vec]))
        layer_1 = tf.add(tf.matmul(x_code_ph, w_1), b_1)
        layer_1 = tf.nn.relu(layer_1)

        w_2 = tf.Variable(tf.random_uniform([self.weight_of_car2vec,3]))
        b_2 = tf.Variable(tf.zeros([3]))
        car_2_vec = tf.add(tf.matmul(layer_1,w_2),b_2)
        car_2_vec = tf.nn.relu(car_2_vec)

        x_all = tf.concat([car_2_vec, x_remain_ph],1)

        w_3 = tf.Variable(tf.truncated_normal([self.num_of_allfeature, self.weight_of_all]))
        b_3 = tf.Variable(tf.zeros([self.weight_of_all]))
        layer_2 = tf.add(tf.matmul(x_all, w_3), b_3)
        layer_2 = tf.nn.relu(layer_2)

        w_o = tf.Variable(tf.random_uniform([self.weight_of_all,1]))
        b_o = tf.Variable(tf.zeros([1]))
        output = tf.add(tf.matmul(layer_2, w_o), b_o)

        cost = tf.reduce_mean((output-y_)**2)
        train = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print(self.x_code)
        print(self.x_remain)
        print(self.y_train)

        for i in range(50):
            for j in range(20):

                x_batch1 = self.x_code[self.num_batch*j:self.num_batch*(j+1),:]
                x_batch2 = self.x_remain[self.num_batch*j:self.num_batch*(j+1),:]
                y_batch = self.y_train[self.num_batch*j:self.num_batch*(j+1),:]
                a=sess.run(train, feed_dict={ x_code_ph:x_batch1 , x_remain_ph:x_batch2 , y_:y_batch})
            
            y_hat = sess.run(output, feed_dict = {x_code_ph : self.x_code, x_remain_ph : self.x_remain, y_ : self.y_train})
            error = self.predict(y_hat, self.y_train)

            print('epoch : ', i+1, 'error : ',error,'%')


"""
plt.scatter(pred, y_test*2000)
plt.plot(pred,pred,'r')
#plt.plot(range(y_test.shape[0]), pred, label="original data")
#plt.plot(range(y_test.shape[0]), y_test*2000, label="predicted data")

plt.xlabel("Predicted price")
plt.ylabel("Actual price")
plt.show()
"""

