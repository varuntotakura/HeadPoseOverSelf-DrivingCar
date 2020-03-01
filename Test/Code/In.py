import numpy as np
import cv2
import time
import win32api as wapi
import os

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def keys_to_output(keys):
    #         w s a d
    output = [0,0,0,0]    
    if 'W' in keys:
        output = [1,0,0,0]
    elif 'S' in keys:
        output = [0,1,0,0]
    elif 'A' in keys:
        output = [0,0,1,0]
    elif 'D' in keys:
        output = [0,0,0,1]    
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name))
else:
    print("File does not exist, starting fresh")
    training_data = []

for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)
    
last_time = time.time()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))
    keys = key_check()
    output = keys_to_output(keys)
    training_data.append([img, output])
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    if len(training_data) % 500 == 0:
        print(len(training_data))
        np.save(file_name, training_data)
        time.sleep(5)
    if cv2.waitKey(30) & 0xff == 'q' == 27:
        break
cap.release()
cv2.destroyAllWindows()
