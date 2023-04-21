from deepface import DeepFace
import os
import cv2

folder = './images/persons/'
sadDir = './images/sad/'
happyDir = './images/happy/'

# Check weather image is sad or happy face using deepface package

def main():
    file_names = os.listdir(folder)

    for name in file_names:
        if name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_url = os.path.join(folder, name)
            print(image_url)
            try:
                face_analysis = DeepFace.analyze(img_path = image_url, actions = [ "emotion"])
                dominant_emotion = next(iter(face_analysis), {}).get('dominant_emotion')
                print(dominant_emotion)
                if dominant_emotion is not None and dominant_emotion in ['angry', 'sad']:
                    img = cv2.imread(image_url, 1)
                    cv2.imwrite(os.path.join(sadDir , name), img)
                
                if dominant_emotion is not None and dominant_emotion  == 'happy':
                    img = cv2.imread(image_url, 1)
                    cv2.imwrite(os.path.join(happyDir , name), img)
            except Exception as e:
                print('#### Error #####')
                print(e)

            



if __name__ == '__main__':
    main()

