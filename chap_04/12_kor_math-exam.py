
import tensorflow as tf

# 입력, 형태만 알려주는 placeholder로 정의
X1 = tf.placeholder(tf.float32, shape=[None])
X2 = tf.placeholder(tf.float32, shape=[None])

# 출력, 형태만 알려주는 placeholder로 정의
Y = tf.placeholder(tf.float32, shape=[None])

# 변수 Weight, Bias 정의. 초기값은 랜덤값
W1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
W2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설식 정의
hypothesis = X1 * W1 + X2 * W2 + b

#cost 함수정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#최적화 함수 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#그래프 실행준비
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#그래프 실행, 500번마다 화면출력
for step in range(5001):
    cost_val, W_val1, W_val2, b_val, _ = \
            sess.run([cost,W1,W2,b,train], \
            feed_dict={X1:[5, 7], X2:[5, 7], Y:[101, 141]})
    if step%500 == 0:
        print(step, cost_val, W_val1, W_val2, b_val)

# 입력 X를 주고 예측 Y를 받아 화면출력
print("예측Y : ", sess.run(hypothesis, feed_dict={X1:[8], X2:[8]}))
