import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

# 读取数据集
train = np.load('trainDistance.npy')
test = np.load('testDistance.npy')

# 把数据分为男女两部分
DR_sex = np.load('DR_sex_map.npy')
DS_sex = np.load('DS_sex_map.npy')
train_pictures_male = DR_sex[DR_sex == 1]
train_pictures_female = DR_sex[DR_sex == 0]

# totals用来存放训练集中男生、女生的总数
totals = {'f': len(train_pictures_male),
          'm': len(train_pictures_female)}
print(totals)

# 第三步 基于朴素贝叶斯的图像分类处理

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB().fit(train, DR_sex)
predictions_labels = clf.predict(test)

print('预测结果:')
print(predictions_labels)

print('算法评价:')
print(classification_report(DS_sex, predictions_labels))

