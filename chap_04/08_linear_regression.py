import tensorflow as tf

# 학습데이터 (X, Y)
x_train = [1, 2, 3, 4]
y_train = [6, 5, 7, 10]

# 변수 선언
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설식 정의 H(x) = Wx+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize, 최적화 함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# 최적값 찾기 반복
for step in range(2000):
   sess.run(train)
   print(step, sess.run(cost), sess.run(W), sess.run(b))

