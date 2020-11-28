from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from imutils.video import WebcamVideoStream
import numpy as np
import cv2

app = Flask(__name__)

model = load_model('maskss.h5')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'with_mask',1:'without_mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

def gen_frames():  
	stream = WebcamVideoStream(src=0).start()
	while True:
		image = stream.read()

		faces=face_clsfr.detectMultiScale(image,1.3,5) 
		for x,y,w,h in faces:

			resized = image[y:y+w,x:x+w]
			resized = cv2.resize(resized,(100,100))
			resized = np.expand_dims(resized,axis=0)
			normalized = resized/255.0
			result = model.predict(normalized) 
			if result[0][0] < 0.5 :
				ans = 0
			else :
				ans = 1
	      
			cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[ans],2)
			cv2.rectangle(image,(x,y-30),(x+w,y),color_dict[ans],-1)
			cv2.putText(image, labels_dict[ans], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


		ret, jpeg = cv2.imencode('.jpg', image)
		data = []
		data.append(jpeg.tobytes())
		frame=data[0]
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



app.run(debug=True)