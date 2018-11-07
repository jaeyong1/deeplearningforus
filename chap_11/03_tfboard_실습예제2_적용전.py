import tensorflow as tf

a = tf.placeholder(tf.float32, name="a")
b = tf.constant(5.0, name="b")
c = tf.add(a, b, name="add_op")

with tf.Session() as sess:
    for step in range(100):
        print(sess.run(c, feed_dict={a: step*1.0}))


'''
확인을 위해서
'''
print("done")
print("run : tensorboard --logdir=/tmp/tensorboard/fori100")
