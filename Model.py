from Header import *


class NN_model(object, x_train_of_code, x_test_of_code, x_train, y_train, y_test, feature_of_code=0, weight_of_car2vec=0):

    def __init__(self):
        
        self.feature_of_code = feature_of_code
        self.weight_of_car2vec = weight_of_car2vec

        self.x_train_of_code = x_train_of_code
        self.x_test_of_code = x_test_of_code
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test



"""
    def car2vec_layer(self):

        x_car2vec = tf.placeholder(tf.float32, [None, self.feature_of_car2vec])

        # car_to_vec_layer

        w_1 = tf.Variable(tf.truncated_normal([self.feature_of_car2vec, self.weight_of_car2vec]))
        b_1 = tf.Variable(tf.zeros([weight_of_car2vec]))
        layer_1 = tf.add(tf.matmul(x, w_1), b_1)
        layer_1 = tf.nn.relu(layer_1)

        w_2 = tf.Variable(tf.random_uniform([weight_of_car2vec,3]))
        b_2 = tf.Variable(tf.zeros([3]))
        car_2_vec = tf.add(tf.matmul(layer_1,w_2),b_2)
        car_2_vec = tf.nn.relu(car_2_vec)

        return car_2_vec

    def Merge(self, ):

        car_2_vec = self.car2vec_layer()
        x_train = np.c_[car_2_vec, self.x_train]

        return x_train

    def Mul_layer(self):

        x_train = tf.placeholder(tf.float32, [None, self.feature_of_

        w_1 = tf.Variable(tf.truncated_normal([self.feature_of_car2vec, self.weight_of_car2vec]))
        b_1 = tf.Variable(tf.zeros([weight_of_car2vec]))
        layer_1 = tf.add(tf.matmul(x, w_1), b_1)
        layer_1 = tf.nn.relu(layer_1)
        
        w_2 = tf.Variable(tf.random_uniform([weight_of_car2vec,3]))
        b_2 = tf.Variable(tf.zeros([3]))
        car_2_vec = tf.add(tf.matmul(layer_1,w_2),b_2)
        car_2_vec = tf.nn.relu(car_2_vec)
         
        return car_2_vec


        # output layer

        w_o = tf.Variable(tf.random_uniform([10,1]))
        b_o = tf.Variable(tf.zeros([1]))
        output = tf.add(tf.matmul(layer_1, w_o), b_o)



cost = tf.reduce_mean((output-y)**2)
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


sess = tf.Session()

sess.run(tf.initialize_all_variables())


"""
for i in range(50):
	for j in range(100):
		
		x_batch = x_train[num_batch*j:num_batch*(j+1),:]
		y_batch = y_train[num_batch*j:num_batch*(j+1),:]
		a=sess.run(train,feed_dict={x:x_train, y:y_train})
		c_t.append(sess.run(cost, feed_dict={x:x_train,y:y_train}))
	
	
	pred = sess.run(output, feed_dict={x:x_test})
	pred = pred*1000
	#print('Epoch :',i+1,'Cost :',np.sqrt(c_t[i]))
	error = sess.run(cost, feed_dict={x:x_test, y:y_test})
	print('Epoch :', i+1, ' Test_Cost :', error)
		
#print(pred)
"""
plt.scatter(pred, y_test*2000)
plt.plot(pred,pred,'r')
#plt.plot(range(y_test.shape[0]), pred, label="original data")
#plt.plot(range(y_test.shape[0]), y_test*2000, label="predicted data")

plt.xlabel("Predicted price")
plt.ylabel("Actual price")
plt.show()
"""

