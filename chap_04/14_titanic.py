"""
Created on Sun May 14 22:50:07 2017

https://www.kaggle.com/c/titanic/data
Titanic: Machine Learning from Disaster

@author: JY
"""
#   'names':   ('PassengerId',     'Survived',  'Pclass',   'Name',  'Sex',      'Age',      'SibSp',    'Parch',   'Ticket', 'Fare'    , 'Cabin', 'Embarked'),                
  #dtype={
  #              'names': ('PassengerId',     'Survived',  'Pclass',   'Name',  'Sex'),
  #              'formats': ( np.int, np.float, np.float,  np.str,  np.str)
  #              },

import numpy as np
import pandas
import tensorflow as tf

#Load CSV file as matrix
train_csv_data = pandas.read_csv('train.csv').as_matrix()
test_csv_data = pandas.read_csv('test.csv').as_matrix()
test_csv_sub = pandas.read_csv('gender_submission.csv').as_matrix()

#MALE -> 1
#FEMALE -> 0
for i in range(len(train_csv_data)):
    #print(train_csv_data[i , 4])
    if train_csv_data[i , 4] == 'male':
        train_csv_data[i, 4] = 1
    else:
        train_csv_data[i, 4] = 0

for i in range(len(test_csv_data)):
    #print(train_csv_data[i , 4])
    if test_csv_data[i , 3] == 'male':
        test_csv_data[i, 3] = 1
    else:
        test_csv_data[i, 3] = 0

#Embarked
# Empty -> 0
# S -> 1
# C -> 2
# Q -> 3
for i in range(len(train_csv_data)):        
    if train_csv_data[i , 11] == 'S':
        train_csv_data[i, 11] = 1
    elif train_csv_data[i , 11] == 'C':
        train_csv_data[i, 11] = 2
    elif train_csv_data[i , 11] == 'Q':
        train_csv_data[i, 11] = 3    
    if np.isnan(train_csv_data[i, 11]):
        train_csv_data[i, 11] = 0               

for i in range(len(test_csv_data)):        
    if test_csv_data[i , 10] == 'S':
        test_csv_data[i, 10] = 1
    elif test_csv_data[i , 10] == 'C':
        test_csv_data[i, 10] = 2
    elif test_csv_data[i , 10] == 'Q':
        test_csv_data[i, 10] = 3    
    if np.isnan(test_csv_data[i, 10]):
        test_csv_data[i, 10] = 0               

X_PassengerData = train_csv_data[:, [2, #Pclass
                           4, #Sex
                           6, #SibSp
                           7, # #Parch
                           11 #Embarked
                           ] ]
Y_Survived = train_csv_data[:, 1:2]

Test_X_PassengerData = test_csv_data[:, [1, #Pclass
                           3, #Sex
                           5, #SibSp
                           6,#, #Parch
                           10 #Embarked
                           ] ]
Test_Y_Survived = test_csv_sub[:, 1:2]

#print(X_PassengerData)
#print(Y_Survived)

#placeholder
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis 
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)


#cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

#Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

previous_cost = 0
#Lauch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):#10000
        cost_val, _ =sess.run([cost, train], feed_dict={X:X_PassengerData, Y:Y_Survived})
    
        if step%500 == 0:
            print("Step =", step, ", Cost: ", cost_val)
           
        #cost 진척이 없으면 조기종료(trick)
        if previous_cost == cost_val:
           print("found best hyphothesis when step : ", step , "\n")
           break
        else:
           previous_cost = cost_val
    
    #가설검증(설명력)
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:X_PassengerData, Y:Y_Survived})
    print("\n Accuracy: " , a)
    print("\n Test CSV runningResult")
    h2,c2,a2 = sess.run([hypothesis, predicted, accuracy], feed_dict={X:Test_X_PassengerData, Y:Test_Y_Survived})
    print("\n Accuracy: " , a2)

print("end~")
