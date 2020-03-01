import numpy as np
import cv2

reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')#haarcascade_eye.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyePair_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leyes = leye_cascade.detectMultiScale(gray)
    reyes = reye_cascade.detectMultiScale(gray)
    #face = face_cascade.detectMultiScale(gray)
    try:
        for (lex,ley,lew,leh),(rex,rey,rew,reh) in leyes, reyes:
            #img = img - img
            leyeCenter = (((lex)+(lex+lew))//2,((ley)+(ley+leh))//2)
            reyeCenter = (((rex)+(rex+rew))//2,((rey)+(rey+reh))//2)
            #cv2.circle(img, leyeCenter, 3, (0, 255, 0), -1)
            #cv2.circle(img, reyeCenter, 3, (0, 255, 0), -1)
            #cv2.rectangle(img,(lex,ley),(lex+lew,ley+leh),(255,255,0),2)
            #cv2.rectangle(img,(rex,rey),(rex+rew,rey+reh),(0,255,0),2)
            #cv2.line(img, leyeCenter, reyeCenter, (255, 255, 255), 2)
            if reyeCenter[1] == leyeCenter[1]:
                text = "-"
            elif reyeCenter[0] < leyeCenter[0]:
                text = "Right"
            elif reyeCenter[0] > leyeCenter[0]:
                text = "Left"
            cv2.putText(img, text, (reyeCenter[0]+100, reyeCenter[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('img',img)
    except:
        pass
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

##import numpy as np
##import cv2
##
##eyePair_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
##eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
##cap = cv2.VideoCapture(0)
##while 1:
##    ret, img = cap.read()
##    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##    eyes = eyePair_cascade.detectMultiScale(gray)
##    eye = eye_cascade.detectMultiScale(gray)
##    for (ex,ey,ew,eh) in eyes:
##        eyeCenter = (((ex)+(ex+ew))//2,((ey)+(ey+eh))//2)
##        eyeCenterLine = ((ex, ((ey)+(ey+eh))//2), (ex+ew,((ey)+(ey+eh))//2))
##        eyeCenterLine1 = (((ex, ((ey)+(ey+eh))//2), (((ex)+(ex+ew))//2,((ey)+(ey+eh))//2)))
##        eyeCenterLine2 = ((((ex)+(ex+ew))//2,((ey)+(ey+eh))//2), (ex+ew,((ey)+(ey+eh))//2))
##        cv2.line(img, eyeCenterLine[0], eyeCenterLine[1], (255, 255, 255), 2)
##        cv2.circle(img, eyeCenter, 3, (0, 255, 0), -1)
##        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
##        for (ex,ey,ew,eh) in eye:
##            eyeCenter = (((ex)+(ex+ew))//2,((ey)+(ey+eh))//2)
##            cv2.circle(img, eyeCenter, 3, (0, 255, 0), -1)
##            if eyeCenter[1] < eyeCenterLine1[0][1] | eyeCenter[1] > eyeCenterLine2[0][1]:
##                tex = "left"
##            elif eyeCenter[1] > eyeCenterLine1[0][1] | eyeCenter[1] < eyeCenterLine2[0][1]:
##                tex = "right"
##            cv2.putText(img, tex, (img.shape[1]-100, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
##    cv2.imshow('img',img)
##    k = cv2.waitKey(30) & 0xff
##    if k == 27:
##        break
##cap.release()
##cv2.destroyAllWindows()
