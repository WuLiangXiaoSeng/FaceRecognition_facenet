# -*- conding:utf-8 -*-

import pickle
import os
import cv2


features_path = "E:/Reference/facenet-new/facedate_person/features_dataset/features.dataset"
face_pic_data = "E:/Reference/facenet-new/facedate_person"



def get_all(path):
    with open(features_path, "rb") as file:
        userdatas = pickle.load(file)
    usernames = []
    for user in userdatas.keys():
        usernames.append(user)

    return usernames, userdatas

def show_all():
    usernames, _  = get_all(features_path)
    for user in usernames:
        # print(user)
        usernames.append(user)
        filename = user  + "_000.jpg"
        path = os.path.join(face_pic_data, filename)
        print(path)
        img = cv2.imread(path)
        cv2.imshow(user, img)
        cv2.waitKey(0)

    print("All Users:", usernames)
    cv2.destroyAllWindows()
    return usernames


def del_user():
    usernames, userdatas = get_all(features_path)
    print(usernames)
    while True:
        username = input("Input the name you want to delete:")
        if username in usernames:
            usernames.remove(username)
            break
        else:
            print("Does not exist! ")
    # 删除
    item = userdatas.pop(username)
    filename = username + "_000.jpg"
    path = os.path.join(face_pic_data, filename)
    print(item)
    with open(features_path, "wb") as file:
        pickle.dump(userdatas, file)

    print("Successfully")


if __name__ == '__main__':
    del_user()
    # usernames, _ = get_all(features_path)
    # print(usernames)