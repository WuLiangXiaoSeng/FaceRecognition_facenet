# -*- conding:utf-8 -*-

import face_colletor
import cv2
import numpy as np
import pickle
from facenet import Facenet
import os
import _thread2
from PIL import Image

import face_recognize
from facenet import Facenet
import os
import pickle
import numpy as np
import cv2
from threading import Thread
from PIL import Image


if __name__ == "__main__":
    while True:
        try:
            ch = int(input("\n         选择：\n1) 录入人脸\n2) 人脸识别\n"))
            if ch not in [1, 2]:
                print("请按照提示输入！！！！！！！！！！")
                continue
            break
        except:
            print("请输入数字！！！！！！！！！！！！")

    if ch == 1:
        face_colletor.get_facedata()
    elif ch == 2:
        face_recognize.test_thread_()
