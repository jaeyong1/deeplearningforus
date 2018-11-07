import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.add(a, b)

with tf.Session() as sess:
    #그래프 연산
    print("c = ", sess.run(c))
