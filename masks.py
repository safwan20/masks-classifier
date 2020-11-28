from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('maskss.h5')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'with_mask',1:'without_mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,gray=source.read()
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:

        resized = gray[y:y+w,x:x+w]
        resized=cv2.resize(resized,(100,100))
        resized = np.expand_dims(resized,axis=0)
        normalized=resized/255.0
        result = model.predict(normalized) 
        if result[0][0] < 0.5 :
            ans = 0
        else :
            ans = 1
      
        cv2.rectangle(gray,(x,y),(x+w,y+h),color_dict[ans],2)
        cv2.rectangle(gray,(x,y-30),(x+w,y),color_dict[ans],-1)
        cv2.putText(
          gray, labels_dict[ans], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',gray)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()