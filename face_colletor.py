# -*- coding:utf-8 -*-

import cv2
import numpy as np
import pickle

from facenet import Facenet

import os
import _thread

from PIL import Image

#---------------------------------------------#
#           人脸采集
#---------------------------------------------#


# 截取后人脸的存放路径
facedata_path = "E:/Reference/facenet-new/facedate_person"
# 人脸特征值存放的文件
features_path = "E:/Reference/facenet-new/facedate_person/features_dataset/features.dataset"
# 人脸检测 Haar级联文件
classfier_paths = ["E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt2.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt_tree.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_default.xml"]


def _thread_saveData(frame, faceRects, person_name, count):
    if len(faceRects) == 0:
        print('Error： 没有检测到人脸')
    elif len(faceRects) > 1:
        print('Error：检测到不止一个人脸')
    else:
        x, y, w, h = faceRects[0]
        image_name = "E:/Reference/facenet-new/facedate_person/%s_%03d.jpg" % (person_name, count)
        image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
        cv2.namedWindow(person_name)
        cv2.imshow(person_name, image)
        if os.path.exists(image_name):
            print('Warning:已经存在该人脸数据')
        print("Notice: 是否保存（y/n)")
        if cv2.waitKey(0) & 0xff == ord('y'):
            cv2.imwrite(image_name, image)
            print(image_name + " has saved.\n")
        cv2.destroyWindow(person_name)



# 识别图像中的人脸，并将人脸部分截取下来，保存
def get_facedata_from_pic(per_name):
    # per_name = input("Please input name:")
    # per_name = "pengyuyan"

    classfier = cv2.CascadeClassifier(classfier_paths[0])
    color = (0, 255, 0)

    while True:
        file_name = input("Please input filename:")
        # file_name = "E:/Reference/facenet-new/PIC/pengyuyan.jpg"
        try:
            # frame_bak = Image.open(file_name)
            frame = cv2.imread(file_name)
        except:
            print("open file error")
            continue
        break

    cv2.namedWindow("frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:

        frame_new = frame.copy()

        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x + w - 15, y + h + 15), color, 2)
        cv2.imshow('frame', frame)

        # 创建线程来供用于选择是否保存。
        try:
            _thread.start_new_thread(_thread_saveData, (frame_new, faceRects, per_name, 0))
        except:
            print('Error: 无法启动线程')
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def get_facedata_from_cap(person_name):
    classfier = cv2.CascadeClassifier(classfier_paths[0])
    color = (0, 255, 0)

    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        key = cv2.waitKey(1) & 0xff
        if not ret:
            print("Capture is not opened!")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:

            # 创建线程来供用于选择是否保存。
            if key == ord('c'):
                # 备份一份数据
                frame_new = frame.copy()
                try:
                    _thread.start_new_thread(_thread_saveData, (frame_new, faceRects, person_name, 0))
                except:
                    print('Error: 无法启动线程')

            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x+w-15, y+h+15), color, 2)
        cv2.imshow('frame', frame)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# 将path目录下所有的图像都进行特征提取并保存
def extract_features(model, features, path):
    filenames = os.listdir(path)

    for filename in filenames:
        if os.path.isfile(os.path.join(path, filename)):
            try:
                image = Image.open(os.path.join(path, filename))
            except:
                print("{} open filed".format(os.path.join(path, filename)))
                break
            person_name = filename.split("_")[0]
            feature = model.get_tensor(image)
            features[person_name] = feature
    print("Successfully extract all features.")

    # 保存
    with open(features_path, "wb") as file:
        pickle.dump(features, file)
    print("Successfully saved features.")




def get_facedata():
    person_name = input("Please input name:")
    if person_name+'_000.jpg' in os.listdir(facedata_path):
        filename = os.path.join(facedata_path, person_name+"_000.jpg")
        try:
            image = cv2.imread(filename)
        except:
            print("{} open failed".format(filename))
        cv2.namedWindow("Show")
        # image = cv2.imread("E:/Reference/facenet-new/facedate_person/wang_000.jpg")
        cv2.imshow("Show", image)
        print("该用户已经存在，是否覆盖（y/n）:")
        key = cv2.waitKey(0) & 0xff
        if key != 'y':
            cv2.destroyWindow("Show")
            return
        else:
            cv2.destroyWindow("Show")
    ch = int(input("从图片添加（1) 打开摄像头添加(2):"))
    if ch == 1:
        get_facedata_from_pic(person_name)
    elif ch == 2:
        get_facedata_from_cap(person_name)


# #########################################
#   以下为测试部分
# #########################################

# 测试get_feace_data函数
def test_getfacedata_from():
    # model = Facenet()

    get_facedata_from_pic()


def test_feature():
    model = Facenet()
    features = {}
    path = "E:/Reference/facenet-new/facedate_person"
    extract_features(model, features, path)


def get_data_fromfile():
    # 读取
    with open(features_path, "rb") as file:
        data = pickle.load(file)
    print(data)

def test_getfacedata():
    model = Facenet()
    get_facedata()
    extract_features(model, {}, facedata_path)

if __name__ == "__main__":
    # test_getfacedata()
    get_data_fromfile()
    # test_feature()

