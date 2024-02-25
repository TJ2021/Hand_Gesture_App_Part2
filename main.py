# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import os
import tensorflow as tf
import frameextractor as fe
import handshape_feature_extractor as hfe
import csv
import re as regex


gestures={"Num0":0,
"Num1":1,
"Num2":2,
"Num3":3,
"Num4":4,
"Num5":5,
"Num6":6,
"Num7":7,
"Num8":8,
"Num9":9,
"FanDown":10,
"FanOn":11,
"FanOff":12,
"FanUp":13,
"LightOff":14,
"LightOn":15,
"SetThermo":16
 }

def extract_feature(location, input_file, mid_frame_counter):
    middle_image = cv2.imread(fe.frameExtractor(location + input_file, location + "frames/", mid_frame_counter),
                              cv2.IMREAD_GRAYSCALE)
    response = hfe.HandShapeFeatureExtractor.extract_feature(hfe.HandShapeFeatureExtractor.get_instance(),
                                                             middle_image)
    return response


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================

trainData = []
trainPath = "traindata/"
count = 0
for file in os.listdir(trainPath):
    if not file.startswith('frames') and not file.startswith('.'):
        trainData.append({file:extract_feature(trainPath, file, count)})
        count = count + 1



# =============================================================================
# Get the penultimate layer for testing data
# =============================================================================
testData =[]
testPath = "test/"
testCount=0
for file in os.listdir(testPath):
    if not file.startswith('frames') and not file.startswith('.'):
        testData.append({file:extract_feature(testPath, file, testCount)})
        testCount = testCount + 1





# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
result = []
for gesture in testData:
    minVal=1
    match =""
    #print(next(iter(gesture.keys())))
    for trainGesture in trainData:
        # cosVal=tf.keras.losses.cosine_similarity(
        # next(iter(trainGesture.values())),
        #     next(iter(gesture.values())),
        #         axis=-1
        #     )

        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        cosVal = cosine_loss( next(iter(trainGesture.values())),next(iter(gesture.values()))).numpy()


        if abs(cosVal)<abs(minVal):
            minVal = cosVal
            match = next(iter(trainGesture.keys())).split("_")[0]
        #print(cosVal, minVal, match)

    #result.append({gesture:gestures[next(iter(trainData[position].keys())).split('_')[0]]})
    #result.append({next(iter(gesture.keys())):gestures[match]})
    result.append(gestures[match])

with open('Result.csv', 'w') as f:
    for line in result:
        f.write(str(line))
        f.write('\n')


