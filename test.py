
import os

# from nets.facenet import facenet
# from facenet import Facenet
from PIL import Image

import cv2

'''
def test_model():
    input_shape = [160,160,3]
    model = facenet(input_shape, len(os.listdir("./datasets")), backbone="inception_resnetv1", mode="train")
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)


def test_faceEncoding():
    model = Facenet()

    # img_path = input("input a image path:")
    input("input anything to continue")
    img_path = 'img/1_001.jpg'
    try:
        image = Image.open(img_path)
    except:
        print('Image_1 Open Error! Try again!')
    
    feceEncoding = model.get_tensor(image)
    print(len(feceEncoding[0]))
    print(feceEncoding)
'''

def get_face_data():
    classfier = cv2.CascadeClassifier('E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt.xml')

    color = (0, 255, 0)
    count = 0
    person_name = input("Please input name:")
    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('cap is not opened!')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        # faceRects = classfier.detectMultiScale(gray, 1.3, 5)
        if len(faceRects) > 0:

            for faceRect in faceRects:
                x, y, w, h = faceRect

                img_name = './facedate_person/%s_%d.jpg' % (person_name, count)

                image = frame[y-10:y+h+10, x-10:x+w+10]
                # cv2.imwrite(img_name, image)

                count += 1
                cv2.rectangle(frame, (x, y), (x+w-15, y+h+15), color, 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    # test_model()
    # test_faceEncoding()
    get_face_data()
