# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:52:56 2018

@author: JY
"""


import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o'] # index를 char로 변환할때 사용
char_dic = {w : i for i, w in enumerate(char_rdic)} # char를 index로 변환할때 사용
ground_truth = [char_dic[c] for c in 'hello']
print ('ground truth \"hello\" : ', ground_truth)
x_data = np.array([[1,0,0,0], # h
                   [0,1,0,0], # e
                   [0,0,1,0], # l
                   [0,0,1,0]], # l
                 dtype = 'f')


x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)
print('x_data : ', x_data)


# Configuration
rnn_size = len(char_dic) # 4
batch_size = 1
output_size = 4



# RNN Model
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
print('rnn_cell : ', rnn_cell)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print('initial_state : ', initial_state)

initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size]) #  위 코드와 같은 결과
print('initial_state_1 : ', initial_state_1)


x_split = tf.split(x_data, len(char_dic), 0) # 가로축으로 4개로 split
print('x_split : ', x_split)
"""
[[1,0,0,0]] # h
[[0,1,0,0]] # e
[[0,0,1,0]] # l
[[0,0,1,0]] # l
"""

#그 다음 output 과 state를 출력
outputs, state = tf.contrib.rnn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)

print ('outputs : ', outputs)
print ('state : ', state)

logits = tf.reshape(tf.concat( outputs, 1), # shape = 1 x 16
                    [-1, rnn_size])        # shape = 4 x 4
logits.get_shape()
"""
[[logit from 1st output],
[logit from 2nd output],
[logit from 3rd output],
[logit from 4th output]]
"""
targets = tf.reshape(ground_truth[1:], [-1]) # a shape of [-1] flattens into 1-D
targets.get_shape()


weights = tf.ones([len(char_dic) * batch_size])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, axis=1))
        print(result, [char_rdic[t] for t in result])
        #print(sess.run(logits))
        #print("outputs:", sess.run(outputs))
        #print("state:", sess.run(state))
#












