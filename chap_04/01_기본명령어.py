# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:15:15 2017

@author: JY
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:01:51 2017

@author: jynote
"""

#from __future__ import division, print_function
#한줄주석
#print("이 줄은 주석 처리 됩니다")

#뒤에 주석처리
print("이 줄은 출력됩니다") #출력문사용법

# '''를 사용한 블럭코멘트
'''
print("여기서부터")
print("여기까지 전부")
print("주석처리 됩니다.")
''' 

# """를 사용한 블럭코멘트
"""
print("여기서부터")
print("여기까지 전부")
print("전부 주석처리 됩니다.")
"""

#주석없는 예시
print("이 줄은 실행됩니다")


# print문의 기본적인 사용
print ("Hello Python. ^-^")

# 정수출력
x = 123;
print ("정수출력: %d" % x) #정수  : 123

# 실수출력
x = 123.456;
print ("실수출력 : %f, %.0f, %.1f" % (x, x, x)) #실수 : 123.456000, 123, 123.5

# 문자열
x = "Hello Python."
y = "Good Morning."
print ("문자열출력 : [%s], [%s] " % (x, x+y))#문자: [Hello Python.], [Hello Python.Good Morning.] 

#print '이코드는 2.7에서는 실행되고 3.5에서는 에러발생합니다.'


#for기본사용법
for x in range(0, 3):
    print ("for연습 : %d" % (x))

#문자열루프
string = "Hello"
for x in string:
    print (x)
    
#배열루프
animals = ["개", "고양이", "곰", "사자", "두더지"]
for x in animals:
    print ("[%s]" % x)

#인덱스값과 함께 루프
animals = ["개", "고양이", "곰", "사자", "두더지"]
for i, x in enumerate(animals):
    print ("[%d][%s]" % (i, x))
    
# if~else 사용법
animals = ["개", "고양이", "곰", "사자", "두더지"]
for x in animals:
    if x in ["개", "고양이"]:
        print ("%s는 귀엽네요." % (x))
    elif x in ["곰", "사자"]:
        print ("%s는 귀엽지만 좀 크네요." % (x))
    else:
        print ("%s는 본적이 없어요." % (x))

    
    
# if~else 기본사용법
a=10
if (a>10):
    print("값은 10보다 큽니다.")
elif (a==10):
    print("값은 10입니다.")
else :
    print("값은 10보다 작습니다.")
    
# if~else 배열사용법
animals = ["개", "고양이", "곰", "사자", "두더지"]
for x in animals:
    if x in ["개", "고양이"]:
        print ("%s는 귀엽네요." % (x))
    elif x in ["곰", "사자"]:
        print ("%s는 귀엽지만 좀 크네요." % (x))
    else:
        print ("%s는 본적이 없어요." % (x))

# while 사용법 (구구단 3단)
count = 1
while count < 10:
    print("3 x %d = %d" % (count, (3*count)))
    count = count +1
    

def sum(a, b): 
    return a + b

# 함수호출과 리턴
a = 10
b = 20
print ("sum : %d + %d = %d" % (a, b, sum(a, b)))

def sum_diff(a, b):
    return a+b, a-b

# 함수 여러값 리턴
a = 100
b = 200
c, d = sum_diff(a, b)
print ("c : %d + %d = %d" % (a, b, c))
print ("d : %d - %d = %d" % (a, b, d))

# 함수리턴값을 따로저장하지 않음
_, e = sum_diff(a, b)
print ("e : %d - %d = %d" % (a, b, e))

# 하나의 변수로 리턴받으면, 정수가 아닌 tuple로 저장됨
f = sum_diff(a, b)
print("f : ", f)

# 리스트
a = [1, 2, 3]
b = [4, 5, 6]
print (a)

# 리스트 합치기
print (a+b)

# 리스트 길이
animals = ["개", "고양이", "곰", "사자", "두더지"]
print(animals)
print ("아이템수 = ", len(animals))
print ("'개'의 수 = ", animals.count("개"))

# 리스트 잘라내기
print("잘라내기[0:2] = " , animals[0:2])
print("잘라내기[:] = " , animals[:])
print("잘라내기[:-1] = " , animals[:-1])

# dictionary 사용예시
dict = {'이름': '바둑이', '견종': '푸들', '나이': 5}
print ("dict['이름'] = ", dict['이름'])

# 변수명의 간략화
import random
weights = {
    'w1': random.randint(1,101),
    'w2': random.randint(1,101),
    'w3': 300
}
print("w1 : ",  weights['w1'])

# class 예시
class Weapon:
    def __init__(self): #생성자
        self.name = "무기"
    def attach(self):
        return "공격"
    def defense(self):
        return "방어"

w = Weapon()
print (w.name, w.attach(), w.defense())


# class 상속
class Gun(Weapon):
    def __init__(self): #생성
        self.name = "총"
    def attach(self, sound="빵빵"):
        return sound
    
g = Gun()
print (g.name, g.attach(), g.defense())
print (g.name, g.attach(sound="빵야빵야"), g.defense())

# 4칸 들여쓰기
a = 10
if a == 10:
    print("good") 
    
# 2칸 들여쓰기
if a >= 2:
  print("morning") 

# 들여쓰기 혼용
if a > 5:
    print("안녕")
  #print("하세요") #IndentationError: unindent does not match any outer indentation level
  