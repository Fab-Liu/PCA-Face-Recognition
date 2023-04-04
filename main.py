import os
import random
import time
import cv2
import tkinter
import custom_tkinter
import tkinter.messagebox

import meta_dataset
import meta_image2mat
import meta_recognitionPCA

import PIL.Image
from PIL import ImageTk, Image
from tkinter import Tk, Label, Button, Canvas

# parameters
recognition_type = 'camera'

# cv2 CascadeClassifier
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_righteye_2splits.xml')

# set custom_tkinter module
custom_tkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
custom_tkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# image path
dir_path = './image_recognize/'
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

# pca prepare process
datasetClass = meta_dataset.DatasetClass('image_train')
image_mat = meta_image2mat.Image2Mat(datasetClass.get_training_image_path(), 100, 100)

pca_matrix = image_mat.get_image_mat()
image_label = datasetClass.get_training_label()
image_width = image_mat.get_image_width()
image_height = image_mat.get_image_height()
category_name = datasetClass.get_category_name()
category_num = datasetClass.get_category_num()

recognitionPCA = meta_recognitionPCA.RecognitionPCA(pca_matrix,
                                                    image_label,
                                                    image_width,
                                                    image_height,
                                                    category_name,
                                                    category_num)
recognitionPCA.calculate_mean_face()
recognitionPCA.reduce_lim()

'''
There are two ways to use the face recognition model:
1. image
    Use the image to image_recognize the face
2. camera
    Use the camera to image_recognize the face
'''


class App(custom_tkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("G0191 Project - Face Recognition with PCA")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = 1100
        window_height = 620
        self.geometry(
            f"{window_width}x{window_height}+{(screen_width - window_width) / 2}+{(screen_height - window_height) / 2}")

        # configure grid layout 
        self.grid_columnconfigure(2, weight=7)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)

        # create sidebar frame with widgets
        self.sidebar_frame = custom_tkinter.CTkFrame(self, corner_radius=0, height=620)
        self.sidebar_frame.grid(column=0, row=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=0)

        # col1 frame
        self.c1_frame = custom_tkinter.CTkFrame(self, bg_color='#242324', width=150)
        self.c1_frame.grid(column=1, row=0)
        self.c1_frame.grid_rowconfigure(5, weight=0)

        self.logo_label = custom_tkinter.CTkLabel(self.sidebar_frame, text="PCA Face\nRecognition",
                                                  font=custom_tkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, padx=0, pady=30)

        # empty
        self.empty_label = custom_tkinter.CTkLabel(self.c1_frame, text='   ', width=150)
        self.empty_label.grid(row=0, pady=38)

        # label
        self.frame_rate = custom_tkinter.CTkLabel(self.c1_frame, text='Frame rate: xxx', width=30,
                                                  font=custom_tkinter.CTkFont(size=16))
        self.frame_rate.grid(row=1, pady=10)
        self.location = custom_tkinter.CTkLabel(self.c1_frame, text='Location: (xx,yy)\n\n\n  ',
                                                font=custom_tkinter.CTkFont(size=16))
        self.location.grid(row=2, pady=10)
        self.tag = custom_tkinter.CTkLabel(self.c1_frame, text='Confidence\nInterval\n \n\n ',
                                           font=custom_tkinter.CTkFont(size=16, weight="bold"))
        self.tag.grid(row=4, pady=10)

        # create textbox
        self.textbox = custom_tkinter.CTkTextbox(self.sidebar_frame, height=335, width=175)
        self.textbox.grid(row=1, padx=15, pady=15)

        # create slider
        self.frame_inner = custom_tkinter.CTkFrame(self.c1_frame, bg_color='#242324')
        self.frame_inner.grid(row=3, column=0)
        self.frame_inner.grid_columnconfigure(2, weight=0)
        self.slider = custom_tkinter.CTkSlider(self.frame_inner, orientation="vertical", height=240)
        self.slider.grid(column=0, padx=5, row=0)
        self.progressbar = custom_tkinter.CTkProgressBar(self.frame_inner, orientation="vertical", height=240)
        self.progressbar.grid(column=1, padx=5, row=0, pady=5)

        self.slider.configure(command=self.progressbar.set)

        for i in get_name_list():
            self.textbox.insert("0.0", i + '\n')

        self.textbox.insert("0.0", "Name list:\n")

        self.camera_main()

    def camera_main(self):
        canvas = Canvas(self, bg='#012145', width=20, height=620)
        canvas.grid(row=0, rowspan=3, column=2, padx=0, pady=0, sticky="nsew")

        camera = cv2.VideoCapture(0)
        count = 0
        while True:
            # read the frame
            ret, frame = camera.read()
            if ret:
                # 抽帧，每隔20帧保存一次
                count += 1
                if count % 1 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray,
                                                          scaleFactor=1.1,
                                                          minNeighbors=5,
                                                          minSize=(30, 30),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)

                    for (x, y, w, h) in faces:
                        # label the face
                        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        f = cv2.resize(gray[y:y + h, x:x + w],
                                       (image_width, image_height),
                                       interpolation=cv2.INTER_AREA)

                        new_coordinate = recognitionPCA.new_coor(f)
                        name = recognitionPCA.recognition(new_coordinate)

                        # CV_CAP_PROP_FPS
                        self.frame_rate.configure(text=f"Frame rate:\n {camera.get(5)}")

                        # cv2.imwrite(face_name, f)
                        count += 1

                        # set info of target
                        self.sidebar_button_1 = custom_tkinter.CTkButton(self.sidebar_frame, text=name)
                        self.sidebar_button_1.grid(row=2, padx=5, pady=70)
                        self.location.configure(text=f"Location:\n ({x}, {y})\n\n  ")

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
                        # left_eye_name = os.path.join(left_eye_path, str(count) + '.jpg')
                        # cv2.imwrite(left_eye_name, f)
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
                        # right_eye_name = os.path.join(right_eye_path, str(count) + '.jpg')
                        # cv2.imwrite(right_eye_name, f)
                        count += 1

                # can not use
                frame = cv2.resize(frame, (920, 620))  # 960x640
                cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
                current_image = Image.fromarray(cvimage)  # 将图像转换成Image对象
                tkImage1 = ImageTk.PhotoImage(image=current_image)
                canvas.create_image(0, 0, anchor='nw', image=tkImage1)
                # make image full of canvas
                canvas.image = tkImage1
                self.update()

                if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
                    break
                    # left_eyes = left_eye_cascade.detectMultiScale(gray, 1.3, 5)
                    # right_eyes = right_eye_cascade.detectMultiScale(gray, 1.3, 5)
            else:
                print('no ret')
                # use break to exit

        camera.release()
        cv2.destroyAllWindows()


def hello_from_GO191_Project():
    print("\n---------------------------------")
    print("hi, this is our G0191 Project \n"
          "for the course of Machine Learning \n"
          "taught by Dr. Wang Han \n"
          "in the Xiamen University Malaysia. \n")
    print("\n---------------------------------")


def get_name_list():
    name_catory_folder = "image_train"
    name_list = []
    # 列出name_category_folder下的所有文件夹
    for name in os.listdir(name_catory_folder):
        if name == '.DS_Store':
            continue
        else:
            name_list.append(name)
    return name_list


if __name__ == "__main__":
    get_name_list()
    app = App()
    app.mainloop()

# python3 G0191-2/main.py
