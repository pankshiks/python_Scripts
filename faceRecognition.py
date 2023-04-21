import cv2
from deepface import DeepFace
import os
import matplotlib.pyplot as plt

path = "./images/persons/"
filesMatch = []

# Check wheather image is same in multiple images
def image_detect(detactImage):
    files = os.listdir(path)
    for img in files:
        print(img)
        if img.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            
            try:
                image_url = os.path.join(path, img)
                result = DeepFace.verify(image_url, detactImage)
                if result["verified"]:
    
                    filesMatch.append(image_url)
                    print("Similarity between", image_url, "is", result["verified"])
                    
            except Exception as e:
                print('#### Error #####')
                print(e)
                continue

def show_images(files):
   for img in files:
        # image_url = os.path.join(path, img)
        imgRead1 = cv2.imread(img)
        plt.imshow(imgRead1)
        plt.show()

if __name__ == '__main__':
    image_url = os.path.join(path, 'image4.jpeg')
    files = image_detect(image_url)
    if len(filesMatch): 
        images = show_images(filesMatch)
