import cv2

img1 = cv2.imread('D:\python\panda.jpg')
img2 = cv2.imread('D:\python\panda1.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create a SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and extract descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Create a FLANN matcher object
matcher = cv2.FlannBasedMatcher()

# Match the descriptors
matches = matcher.match(des1, des2)

# Sort the matches by score
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top matches
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)

# Show the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if the images contain the same object
if len(matches) > 10:
    print("The images contain the same object")
else:
    print("The images do not contain the same object")