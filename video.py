# -*- conding:utf-8 -*-

import cv2

def test_video():
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)


    while True:
        ret, img = cap.read()


        if ret:
            cv2.imshow("input", img)
            key = cv2.waitKey(0) & 0xff
            if key == 27:
                break
            elif key == ord('s'):
                value = int(input("亮度："))
                cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
        else:
            print("ret")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_video()