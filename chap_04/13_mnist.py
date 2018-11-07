
import tensorflow as tf

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 변수들을 설정한다.
X = tf.placeholder(tf.float32, [None, 784]) # mnist 이미지데이터 형태는 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9 숫자분류 = > 10 classes

#Logistic Classifier (Linear Classifier)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
logit_y = tf.matmul(X, W) + b

# softmax와 cross-entropy 모델을 설정한다.
softmax_y = tf.nn.softmax(logit_y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(softmax_y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 경사하강법으로 모델을 학습한다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) #배치크기는 100
  sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

# 학습된 모델이 얼마나 정확한지를 계산하고 출력한다.
correct_prediction = tf.equal(tf.argmax(softmax_y,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("정확도 : ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

'''
#실행결과
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
정확도 :  0.9092
'''