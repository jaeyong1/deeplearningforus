# -*- coding: utf-8 -*-
"""
비행기무게에 따른 이륙거리 예측하기

Created on Fri Apr 20 2018

@author: jynote
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#데이터셋을 섞음(train/test data set)
def shuffle_data(x_train,y_train):
  temp_index = np.arange(len(x_train))

  #Random suffle index
  np.random.shuffle(temp_index)

  #Re-arrange x and y data with random shuffle index
  x_temp = np.zeros(x_train.shape)
  y_temp = np.zeros(y_train.shape)
  x_temp = x_train[temp_index]
  y_temp = y_train[temp_index]        

  return x_temp, y_temp

#0에서 1사이 값으로 정규화
def minmax_normalize(x):
  xmax, xmin = x.max(), x.min()
  return (x - xmin) / (xmax - xmin)

#정규화 후 realx에 해당하는 정규값 리턴
def minmax_get_norm(realx, arrx):
  xmax, xmin = arrx.max(), arrx.min()  
  normx = (realx - xmin) / (xmax - xmin)
  return normx    
    
#0에서 1사이 값을 실제 값으로 역정규값 리턴
def minmax_get_denorm(normx, arrx):
  xmax, xmin = arrx.max(), arrx.min()
  realx = normx * (xmax - xmin) + xmin
  return realx
  

def main():
  traincsvdata = np.loadtxt('trainset.csv', unpack=True, delimiter=',', skiprows=1)
  num_points = len(traincsvdata[0]) 
  #print(traincsvdata)
  print("points : ", num_points)
  
  x1_data = traincsvdata[0]  # Speed(km/h) : [310. 312. 314. 319. 
  x2_data = traincsvdata[1] # Weight(ton) : [261. 221. 261. 274. 221....    
  y_data = traincsvdata[2]  # Distance(m) : [1814. 1632. 1981. 2163. 1733.

  #빨간색(m)에 둥근점(o)로 시각화
  plt.plot(x1_data, y_data, 'mo')  
  plt.suptitle('Traing set(x1)', fontsize=16)  
  plt.xlabel('speed to take off')
  plt.ylabel('distance')
  plt.show()

  #파란색(b)에 둥근점(o)로 시각화
  plt.plot(x2_data, y_data, 'bo')  
  plt.suptitle('Traing set(x2)', fontsize=16)  
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()

  
  #데이터 정규화를 수행. 0~1사이 값으로 변환
  x1_data = minmax_normalize(x1_data)
  x2_data = minmax_normalize(x2_data)
  y_data = minmax_normalize(y_data)
    
  #x_data생성
  x_data = [[item for item in x1_data], [item for item in x2_data]] # [[0.6666666666666666, 0.7333333333333333, 0.8, 0.9666666666666667, ..
  x_data = np.reshape(x_data, 600, order='F') # [0.66666667 0.56666667 0.73333333  ..
  x_data = np.reshape(x_data, (-1,2)) # [[0.66666667 0.56666667] [0.73333333 0.12222222] ..
  
  #y_data reshape
  y_data = np.reshape(y_data, [len(y_data),1]) # [[0.56734694]  [0.38163265]  [0.7377551 ]  [0.92346939]
  
  '''
  #ref)
  lst_speed = [item[0] for item in x_data]
  lst_weight = [item[1] for item in x_data]
   
  #빨간색(m)에 둥근점(o)로 시각화
  print("NORM")
  plt.plot(lst_speed, y_data, 'mo')  
  plt.xlabel('speed to take off')
  plt.ylabel('distance')
  plt.show()

  #빨간색(b)에 둥근점(o)로 시각화
  plt.plot(lst_weight, y_data, 'bo')  
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()
  '''

  #배치 수행
  BATCH_SIZE = 5
  BATCH_NUM = int(len(x1_data)/BATCH_SIZE)
  
  
  #총 개수는 정해지지 않았고 1개씩 들어가는 Placeholder 생성
  input_data = tf.placeholder(tf.float32, shape=[None,2])  
  output_data = tf.placeholder(tf.float32, shape=[None,1])

  #레이어간 Weight 정의후 랜덤값으로 초기화. 그림에서는 선으로 표시.
  W1 = tf.Variable(tf.random_uniform([2,5], 0.0, 1.0))
  W2 = tf.Variable(tf.random_uniform([5,3], 0.0, 1.0))
  W_out = tf.Variable(tf.random_uniform([3,1], 0.0, 1.0))

  #레이어의 노드가 하는 계산. 이전노드와 현재노드의 곱셈. 비선형함수로 sigmoid 추가.
  hidden1 = tf.nn.sigmoid(tf.matmul(input_data,W1))
  hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2))
  output = tf.matmul(hidden2, W_out)

  #비용함수, 최적화함수, train 정의
  loss = tf.reduce_mean(tf.square(output-output_data))
  optimizer = tf.train.AdamOptimizer(0.01)
  train = optimizer.minimize(loss)

  #변수(Variable) 사용준비
  init = tf.global_variables_initializer()

  #세션 열고 init 실행
  sess= tf.Session()
  sess.run(init)


  #학습을 반복하며 값 업데이트
  for step in range(1000):
    index = 0    

    #매번 데이터셋을 섞음
    x_data, y_data = shuffle_data(x_data, y_data)

    #배치크기만큼 학습을 진행
    for batch_iter in range(BATCH_NUM-1):
        feed_dict = {input_data: x_data[index:index+BATCH_SIZE], output_data: y_data[index:index+BATCH_SIZE]}
        sess.run(train, feed_dict = feed_dict)
        index += BATCH_SIZE
        
    #화면에 학습진행상태 출력
    if (step%200==0): 
        print("Step=%5d, Loss Value=%f" %(step, sess.run(loss, feed_dict = feed_dict)))      


  
  #학습이 끝난후 테스트 데이터 입력해봄
  print("# 학습완료. 임의값으로 이륙거리추정#")
  arr_ask_x = [[290, 210], #expect:1314
  [320, 210], #expect:1600
  [300, 300], #expect:2009
  [320, 300] #expect:2286
  ] 
  
  for i in range(len(arr_ask_x)):
      ask_x = [arr_ask_x[i]] #테스트용
      ask_norm_x = [[minmax_get_norm(ask_x[0][0], traincsvdata[0]),  minmax_get_norm(ask_x[0][1], traincsvdata[1])]]
      answer_norm_y = sess.run(output, feed_dict={input_data: ask_norm_x})
      answer_y = minmax_get_denorm(answer_norm_y, traincsvdata[2])
      print("이륙거리계산) 이륙속도(x1): ", ask_x[0][0], "km/h, ",
           "비행기무게(x2): ", ask_x[0][1], "ton, ",
           "이륙거리(y): ", answer_y[0][0], "m")
   
      
      
  #테스트셋을 활용한 결과확인  
  print("\n\n\n")
  print("## Test Set 검증결과 그래프 ##")
        
  #테스트셋 파일읽음
  test_csv_x_data = np.loadtxt('testset_x.csv', unpack=True, delimiter=',', skiprows=1)
  test_csv_y_data = np.loadtxt('testset_y.csv', unpack=True, delimiter=',', skiprows=1)
  test_x1_data = test_csv_x_data[0]  # Speed(km/h) : [319. 316. 317. 314. ...
  test_x2_data = test_csv_x_data[1] # Weight(ton) : [298. 298. 293. 295. ...  
  
  #테스트셋 정규화 진행  
  test_x1_data = minmax_normalize(test_x1_data) # [1. 0.89655172 0.93103448 ...
  test_x2_data = minmax_normalize(test_x2_data) # [0.97752809 0.97752809 0.92134831  ...
  test_y_data = minmax_normalize(test_csv_y_data) # [1. 0.95138889 0.92476852 0.89351852

  
  #testset의 x_data생성
  test_x_data = [[item for item in test_x1_data], [item for item in test_x2_data]]
  test_x_data = np.reshape(test_x_data, len(test_x1_data)*2, order='F') 
  test_x_data = np.reshape(test_x_data, (-1,2)) # [[1. 0.97752809] [0.89655172 0.97752809] [0.93103448 0.92134831]...
   
  #테스트셋 CVS파일 : 빨간색(m)에 둥근점(o)로 시각화
  #저자가 임의로 내림차순으로 정렬해 놓음.
  plt.plot(list(range(len(test_csv_y_data))), test_csv_y_data, 'mo')
  
  #예측데이터 : 검은색(k) 별표(*)로 시각화    
  feed_dict = {input_data: test_x_data } 
  test_pred_y_data = minmax_get_denorm(sess.run(output, feed_dict=feed_dict),  traincsvdata[2]) #학습데이터 기준으로 denormalization 수행 
  plt.plot(list(range(len(test_csv_y_data))), test_pred_y_data, 'k*')
  
  #그래프 표시
  plt.suptitle('Test Result', fontsize=16)  
  plt.xlabel('index(x1,x2)')
  plt.ylabel('distance')
  plt.show()
  

#main함수
if __name__ == "__main__":
  main()
