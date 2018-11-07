import tensorflow as tf

# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a *b
print("c = ", c)

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print("sess.run(c) = ", sess.run(c))
