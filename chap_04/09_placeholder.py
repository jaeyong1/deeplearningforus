import tensorflow as tf 

# Build a graph.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.multiply(a, b)
print("node c = ", c)

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print("run c = ", sess.run(c, feed_dict={a: 3., b: 4.}))

'''
# 실행결과
node c =  Tensor("Mul:0", dtype=float32)
run c =  12.0
''''

