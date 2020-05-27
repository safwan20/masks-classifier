import os
import cv2

path = '/home/safwan/Documents/observations-master/experiements/dest_folder/val/with_mask'
img_names = os.listdir(path)

cnt = 1
for i in img_names :
    full = '/home/safwan/Documents/observations-master/experiements/dest_folder/val/with_mask/' + i
    img = cv2.imread(full)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    gray = cv2.resize(gray,(100,100))
    cv2.imwrite('/home/safwan/Desktop/classify_masks/n/val/with_mask/' + str(cnt) + '.jpg' , gray)
    cnt+=1