import face_recognition
import cv2

# find specific image from group image and circle on image
group_photo = face_recognition.load_image_file("./images/group1.jpg")

known_face = face_recognition.load_image_file("./images/image2.jpg")
face_locations = face_recognition.face_locations(group_photo)
findImage = face_recognition.face_encodings(known_face)
if len(findImage):
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = group_photo[top:bottom, left:right]
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding):
            face_distance = face_recognition.compare_faces([findImage[0]], face_encoding[0], tolerance=0.5)[0]
            if face_distance == True:
            
                cv2.rectangle(group_photo, (left, top), (right, bottom), (0, 255, 0),  2)

                # Display the group photo with detected faces
                cv2.imshow("Group Photo", group_photo)

                cv2.waitKey(0)
            # else:
                # print('No Image found')






