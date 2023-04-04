import os

import cv2

'''
There are two ways to use the face recognition model:
1. image
    Use the image to image_recognize the face
2. camera
    Use the camera to image_recognize the face
'''

# parameters
recognition_type = 'camera'

# image path
dir_path = '../image_recognize/'
face_path = os.path.join(dir_path, 'face')
left_eye_path = os.path.join(dir_path, 'left_eye')
right_eye_path = os.path.join(dir_path, 'right_eye')

# create the path
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
if not os.path.exists(face_path):
    os.makedirs(face_path)
if not os.path.exists(left_eye_path):
    os.makedirs(left_eye_path)
if not os.path.exists(right_eye_path):
    os.makedirs(right_eye_path)

# print the path
print(f"face_path: {face_path}")
print(f"left_eye_path: {left_eye_path}")
print(f"right_eye_path: {right_eye_path}")

# cv2 CascadeClassifier
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_righteye_2splits.xml')


def hello_from_GO191_Project():
    print("hi, this is our G0191 Project "
          "for the course of Machine Learning"
          "taught by Dr. Wang Han"
          "in the Xiamen University Malaysia.")
    print("\n---------------------------------")


if __name__ == "__main__":
    hello_from_GO191_Project()

    # image
    if recognition_type == 'image':
        pass

    # camera
    if recognition_type == 'camera':
        camera = cv2.VideoCapture(0)
        count = 0
        while True:
            # read the frame
            ret, frame = camera.read()
            if ret:
                # 抽帧，每隔20帧保存一次
                count += 1
                if count % 20 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,
                                                          scaleFactor=1.1,
                                                          minNeighbors=5,
                                                          minSize=(30, 30),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)

                    for (x, y, w, h) in faces:
                        # label the face
                        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                        # save
                        face_name = os.path.join(face_path, str(count) + '.jpg')
                        cv2.imwrite(face_name, f)
                        count += 1

                    left_eye = left_eye_cascade.detectMultiScale(gray,
                                                                 scaleFactor=1.1,
                                                                 minNeighbors=5,
                                                                 minSize=(30, 30),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)

                    for (x, y, w, h) in left_eye:
                        # label the face
                        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                        # save
                        left_eye_name = os.path.join(left_eye_path, str(count) + '.jpg')
                        cv2.imwrite(left_eye_name, f)
                        count += 1

                    right_eye = right_eye_cascade.detectMultiScale(gray,
                                                                   scaleFactor=1.1,
                                                                   minNeighbors=5,
                                                                   minSize=(30, 30),
                                                                   flags=cv2.CASCADE_SCALE_IMAGE)

                    for (x, y, w, h) in right_eye:
                        # label the face
                        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                        # save
                        right_eye_name = os.path.join(right_eye_path, str(count) + '.jpg')
                        cv2.imwrite(right_eye_name, f)
                        count += 1

                    # left_eyes = left_eye_cascade.detectMultiScale(gray, 1.3, 5)
                    # right_eyes = right_eye_cascade.detectMultiScale(gray, 1.3, 5)

            else:
                print('no ret')
                # use break to exit
                if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
                    break
