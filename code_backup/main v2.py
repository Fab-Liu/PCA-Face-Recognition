import os
import cv2
import meta_dataset
import meta_image2mat
import meta_recognitionPCA
import PIL.Image
from PIL import ImageTk,Image
from tkinter import Tk, Label, Button, Canvas

'''
There are two ways to use the face recognition model:
1. image
    Use the image to image_recognize the face
2. camera
    Use the camera to image_recognize the face
'''

#init the window
win= Tk()
win.title('Fianl Assignment')
win.geometry('960x640')
win.configure(background='#012145')

#text
blank = Label(win,width=0,height=0,background='#012145',text='   ',font=('times',26))
tmpTxt = 'Please insert your data first!'
tmp2 = '''Make sure your data is in the database before starting detecting!!'''
txt = Label(win,width=0,height=0,background='#012145',text=tmpTxt,font=('times',26))
blank.pack(pady=60)
txt.pack(pady=0)
txt2 = Label(win,width=0,height=0,background='#012145',text=tmp2,font=('times',18))
txt2.pack(pady=40)

def clean1():
    blank.pack_forget()
    txt.pack_forget() 
    txt2.pack_forget()  
    button1.pack_forget() 
    button2.pack_forget() 
    leader.place_forget() 
    mem1.place_forget() 
    mem2.place_forget() 
    mem3.place_forget() 
    # button = Button(win,width=15,height=3,text='Exist',background='#013569',highlightbackground='#013569',highlightcolor='#0b4178')
    # button.place(x=800,y=300)
    # button.pack(pady=0)


def drawCan():
    #绘制视频画布
    clean1()
    title_name = Label(win,width=0,height=0,background='#012145',text="Name: ",font=('times',20))
    title_gender = Label(win,width=0,height=0,background='#012145',text="Gender: ",font=('times',20))
    title_status = Label(win,width=0,height=0,background='#012145',text="Status: ",font=('times',20))

    title_name.place(x=180,y=550)
    # title_gender.place(x=450,y=550)
    # title_status.place(x=720,y=550)

# parameters
recognition_type = 'camera'

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

# cv2 CascadeClassifier
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_righteye_2splits.xml')

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


def hello_from_GO191_Project():
    print("\n---------------------------------")
    print("hi, this is our G0191 Project \n"
          "for the course of Machine Learning \n"
          "taught by Dr. Wang Han \n"
          "in the Xiamen University Malaysia. \n")
    print("\n---------------------------------")


def camera_main():
    canvas = Canvas(win,bg = '#012145',width = 960,height = 500 )
    canvas.pack()
    drawCan()

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

                        # save
                        face_name = os.path.join(face_path, str(count) + '.jpg')
                        # cv2.imwrite(face_name, f)
                        count += 1
                        #set info of target
                        txt_name = Label(win,width=15,height=0,background='#012145',text=name,font=('times',20))
                        txt_name.place(x=260,y=550)

                    '''
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
                    '''
                # frame = cv2.resize(frame, (1080, 568))
                # cv2.imshow('frame', frame)
                frame = cv2.resize(frame, (960, 620))# 960x640
                cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                current_image = Image.fromarray(cvimage)#将图像转换成Image对象
                tkImage1 = ImageTk.PhotoImage(image=current_image)
                canvas.create_image(0,0,anchor = 'nw',image = tkImage1)
                win.update()

                if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
                    break
                    # left_eyes = left_eye_cascade.detectMultiScale(gray, 1.3, 5)
                    # right_eyes = right_eye_cascade.detectMultiScale(gray, 1.3, 5)
            else:
                print('no ret')
                # use break to exit



        camera.release()
        cv2.destroyAllWindows()
#按钮
button1 = Button(win,width=20,height=3,text='Collect face information',background='#013569',highlightbackground='#013569',highlightcolor='#0b4178')
#button1.place(x=350,y=280)
button1.pack(pady=20)

button2 = Button(win,width=20,height=3,text='Start detecting',command=camera_main,background='#013569',highlightbackground='#013569',highlightcolor='#0b4178')
#button2.place(x=350,y=400)
button2.pack(pady=30)

leader = Label(win,width=0,height=0,background='#012145',text='Leader:   Liu Aofan',font=('times',18))
leader.place(x=700,y=520)

mem1 = Label(win,width=0,height=0,background='#012145',text='Member: Tan Qianqian',font=('times',18))
mem1.place(x=700,y=540)
mem2 = Label(win,width=0,height=0,background='#012145',text='Hong Chang',font=('times',18))
mem2.place(x=770,y=560)
mem3 = Label(win,width=0,height=0,background='#012145',text='Bai Rui',font=('times',18))
mem3.place(x=770,y=580)

win.mainloop()

#python3 G0191-2/main.py