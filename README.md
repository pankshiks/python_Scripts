Packages:
1. DeepFace (facial recognition library)
2. cv2
3. fer (Facial expression recognition)


1. In main.py i used the DeepFace package to get image emotion and this package returned results ['age', 'gender', 'race', 'emotion', 'dominant_emotion'] of the image
and got dominant_emotion from the result 

2. In app.py I used fer package in this they return the emotion of image in persertage like:
	'emotions': {'angry': 0.02, 'disgust': 0.0, 'fear': 0.05, 'happy': 0.16, 'neutral': 0.09, 'sad': 0.27, 'surprise': 0.41}
and for get highest persentage of emotion i use top_emotion fer function

3. cv2 package used for read and write image in folder

