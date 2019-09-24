# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:51:47 2019

@author: Administrator
"""

from PIL import Image
import numpy
import numpy as np
from keras.models import load_model
import math

#神经网络模型加载
model = load_model('C:\\Users\\Administrator\\Desktop\\ocr\\classifier\\model.h5')#（修改）模型文件的位置
#图像读取，处理为适应模型的矩阵
im = Image.open("C:\\Users\\Administrator\\Desktop\\ocr\\code1.jpg","r")#（修改）用于测试的图像文件路径，可按需求替换
im = im.convert('L') 
[m,n] = im.size
height = n
width = round(n/2)
step = round(n/14)
step_num = int((m-n/2)/step+1)
x0 = 0
result = numpy.zeros((step_num,1))
for i in range(step_num):
    x0 = x0+step
    region = (x0,0,x0+width,n)
    patch = im.crop(region)
    patch = patch.resize((32,32),Image.BICUBIC)
    mat = numpy.asarray(patch).T
    matrix = mat.reshape(1, 32, 32, 1)
    matrix = matrix.astype('float32')
    matrix /= 255


    #输出预测值
    test_predict=(np.asarray(model.predict(matrix))).round()
    if sum(sum(test_predict)) == 0:
        result[i] = -1
    elif sum(sum(test_predict)) == 10:
        result[i] = -2
    else:
        result[i] = (np.argmax(test_predict))
            
#去除两端的无数字空白部分
detector_step = math.ceil(m/(2*n))
sparsity = numpy.zeros(np.size(result)-detector_step+1)
for j in range(np.size(sparsity)):
    detector = result[j:j+detector_step]
    sparsity[j] = ((sum(sum(detector==-2))/detector_step)>0.5)
for k in range(np.size(result)-detector_step):
    if sparsity[k+1]<sparsity[k]:
        result = result[k+detector_step:]
        break
for k in range(np.size(result)-detector_step):
    if sparsity[-k]>sparsity[-k-1]:
        result = result[:-k-detector_step+1]
        break
        
#将各数字分隔开
for i in range(np.size(result)-1):
    if (i!=0 and result[i]<0 and result[i-1]==result[i+1]):
        result[i] = result[i-1]
x=np.array([result[1]])
for i in range(np.size(result)-1):
    if result[i]!=result[i+1] or result[i+1]>=0:
        x = np.append(x,result[i+1])
        
        
        
#识别各分割区间内最可能的数字，保存
one_digit = np.array([])
counts = np.zeros(11)#用来保存11位手机号码
t = 0
for i in range(np.size(x)):
    if i==0 and x[i]<0:
        one_digit = one_digit[1:]
    elif i==(np.size(x)-1) and x[i]>0:
        one_digit = np.append(one_digit,-1)
    elif x[i]>=0:
        one_digit = np.append(one_digit,int(x[i]))
        
    else:
        one_digit = one_digit.astype('int64')

        counts[t] = np.argmax(np.bincount(one_digit))
        t=t+1
        one_digit = np.array([])

print(counts)

            
        
        













