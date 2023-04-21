from fer import FER
import os
import cv2

folder = './images/persons/'
sadDir = './images/sad/'
happyDir = './images/happy/'

# Check weather image is sad or happy face using fer package
def main():
    file_names = os.listdir(folder)

    for name in file_names:
        if name.endswith(('.jpg', '.jpeg', '.png')):
            image_url = os.path.join(folder, name)

            emo_detector = FER()
            # Capture all the emotions on the image
            try: 
                emotion, score = emo_detector.top_emotion(image_url)

                if emotion is not None and emotion in ['angry', 'sad']:
                    img = cv2.imread(image_url, 1)
                    cv2.imwrite(os.path.join(sadDir , name), img)
                
                elif emotion is not None and emotion  == 'happy':
                    img = cv2.imread(image_url, 1)
                    cv2.imwrite(os.path.join(happyDir , name), img)

                elif emotion is None:
                    print('Face not detacted')
            except:
                print('#### Error #####')
                print('Something went wrong')


if __name__ == '__main__':
    main()

