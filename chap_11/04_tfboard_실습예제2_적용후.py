import tensorflow as tf

a = tf.placeholder(tf.float32, name="a")
b = tf.constant(5.0, name="b")
c = tf.add(a, b, name="add_op")

# step 1: scalars 기록할 node 선택
tf.summary.scalar('add_result_c', c)

# step 2: summary 통합
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # step 3: writer 생성
    writer = tf.summary.FileWriter('/tmp/tensorboard/fori100', sess.graph)

    for step in range(100):
         # step 4: 그래프 실행, Summary 정보 로그 추가
        summary = sess.run(merged, feed_dict={a: step*1.0})
        writer.add_summary(summary, step)
print("Done")

'''
확인을 위해서
'''
print("run : tensorboard --logdir=/tmp/tensorboard/fori100")
