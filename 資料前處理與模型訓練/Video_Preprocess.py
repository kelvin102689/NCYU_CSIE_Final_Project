from facenet_pytorch import MTCNN      # *******
from matplotlib import pyplot as plt
import cv2
import math
import numpy as np
import time
import imutils
import os
from PIL import Image                # *******
import threading
import torch
from pathlib import Path
size=(600,600)
TargetFramNum = 150


def video_to_frame(FileRoot):
    start = time.time()
    pathlist = Path(FileRoot).glob('**/*.mp4')
    i=0
    for path in pathlist:
        video_name = path.name
        #print(video_name)

        folder_name = video_name.split('.')[0]
        # 建該影片預處理的資料夾
        create_folder("root", folder_name)
        #差值取偵
        img_list = extract_keyframe("./" + FileRoot +"/" + video_name , TargetFramNum)
        #代表該影片總偵數 < 目標偵數 =>直接放棄該影片
        if img_list==[]:
            continue
        #人臉偵測
        face_detect(video_name, img_list)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(folder_name +'_original' + str(i) + '.mp4', fourcc, 30, size)
        for k in range(0, 150):
            # imagename = './preprocess/id0_000'+str(i)+'/original/id0_000'+str(i)+'_original' + str(
            imagename = './preprocess/' + folder_name + '/original/' + folder_name + '_original' + str(
                k) + '.jpg'  # 'output'+str(i)+'.jpg' 我改過了
            img = cv2.imread(imagename)
            if img is not None:
                out.write(img)
        out.release()
        cv2.destroyAllWindows()
        # color_temperature
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(folder_name +'_color_temperature' + str(i) + '.mp4', fourcc, 30, size)
        for k in range(0, 150):
            imagename = './preprocess/' + folder_name + '/color_temperature/' + folder_name + '_color_temperature' + str(
                k) + '.jpg'  # 'output'+str(i)+'.jpg' 我改過了
            img = cv2.imread(imagename)
            if img is not None:
                out.write(img)
        out.release()
        cv2.destroyAllWindows()
        # contrast
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(folder_name +'_contrast' + str(i) + '.mp4', fourcc, 30, size)
        for k in range(0, 150):
            imagename = './preprocess/' + folder_name + '/contrast/' + folder_name + '_contrast' + str(
                k) + '.jpg'  # 'output'+str(i)+'.jpg' 我改過了
            img = cv2.imread(imagename)
            if img is not None:
                out.write(img)
        out.release()
        cv2.destroyAllWindows()

    end = time.time()
    print("執行時間%d" % (end - start))


def frame_to_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputV.mp4', fourcc, 15, size)
    for i in range(0,60):
        imagename = 'output'+str(i)+'.jpg'
        img = cv2.imread(imagename)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()

def face_detect(video_name , img_list):
    video_name = video_name.split('.')[0]
    create_folder("original" , video_name)
    filename = video_name + '_original'
    parent_dir = "./preprocess/" + video_name +"/original/"
    mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda')
    index = 0
    crop_imgs = []
    #print(len(img_list))
    for img in img_list:
        # 縮小圖片
        img = imutils.resize(img, height=600,width=600)

        #OpenCV預設讀取channel為BGR  PIL為RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #將 NumPy 陣列轉換為 PIL 影象
        img = Image.fromarray(img)

        boxes, probs = mtcnn.detect(img)
        #有bounding box再做後續處理
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                #print(len(boxes))
                #print(box)
                print("%s Face Detect Precision : %f" % (video_name + str(index), prob))
                #偵測到的box人臉機率大於0.6才進行儲存
                if prob > 0.6:
                    for i in range(0, 4):  # box[0],box[1]左上角的x,y座標  box[2],box[3]右下角x,y座標
                        if i == 0 or i == 1:
                            box[i] -= 3
                        else:
                            box[i] += 3
                    try:
                        Path = parent_dir + filename + str(index)+'.jpg'
                        img = img.crop(box)
                        img = img.resize((600, 600))
                        img.save(Path)
                        img = cv2.imread(Path)  #可再優化(直接將img轉cv2格式)
                        crop_imgs.append(img)
                        index += 1
                    except Exception as e:
                        print(e)
                        print("儲存或讀取人像圖片時發生錯誤!!")
                else:
                    print("%s Face Detect Precision is too low ---> abort!")

    t1 = threading.Thread(target=Modify_Contrast, args=(video_name, crop_imgs,))  # 建立執行緒
    t2 = threading.Thread(target=modify_color_temperature, args=(video_name, crop_imgs,))  # 建立執行緒
    #Modify_Contrast(video_name , crop_imgs)
    #modify_color_temperature(video_name, crop_imgs)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("%s preprocessing had completed ! Total %d frames" %(video_name , len(crop_imgs)))

def Modify_Contrast(video_name , crop_imgs):
    create_folder("contrast" , video_name)
    index = 0
    for img in crop_imgs:
        filename = "./preprocess/" + video_name + "/contrast/" + video_name + "_contrast" +str(index) + ".jpg"

        brightness = 0
        contrast = 25  # - 減少對比度/+ 增加對比度

        B = brightness / 255.0
        c = contrast / 255.0
        k = math.tan((45 + 44 * c) / 180 * math.pi)

        img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

        # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
        img = np.clip(img, 0, 255).astype(np.uint8)
        if cv2.imwrite(filename, img):
            index+=1



def modify_color_temperature(video_name , crop_imgs):
    create_folder("color_temperature", video_name)
    index = 0
    for img in crop_imgs:
        filename = "./preprocess/" + video_name + "/color_temperature/" + video_name + "_color_temperature" +str(index) + ".jpg"
        # 1.計算三個通道的平均值，並依照平均值調整色調
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]
        # 調整色調請調整這邊~~
        # 白平衡 -> 三個值變化相同
        # 冷色調(增加b分量) -> 除了b之外都增加
        # 暖色調(增加r分量) -> 除了r之外都增加
        bAve = cv2.mean(imgB)[0]
        gAve = cv2.mean(imgG)[0] + 10
        rAve = cv2.mean(imgR)[0] + 10
        aveGray = (int)(bAve + gAve + rAve) / 3

        # 2. 計算各通道增益係數，並使用此係數計算結果
        bCoef = aveGray / bAve
        gCoef = aveGray / gAve
        rCoef = aveGray / rAve
        imgB = np.floor((imgB * bCoef))  # 向下取整
        imgG = np.floor((imgG * gCoef))
        imgR = np.floor((imgR * rCoef))
        imgb = imgB
        imgb[imgb > 255] = 255

        imgg = imgG
        imgg[imgg > 255] = 255

        imgr = imgR
        imgr[imgr > 255] = 255

        cold_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8)

        if cv2.imwrite(filename, cold_rgb):
            index+=1

def create_folder(type , folder_name):
    folder = None
    if type == "root":
        parent_dir = "./preprocess"
        folder = os.path.join(parent_dir, folder_name)
    else:
        parent_dir = "./preprocess/" + folder_name
        if type == "original":
            folder = os.path.join(parent_dir, "original")
        elif  type == "contrast":
            folder = os.path.join(parent_dir, "contrast")
        elif type == "color_temperature":
            folder = os.path.join(parent_dir, "color_temperature")
    # 如果該影片預處理的資料夾未存在則創造
    if not os.path.exists(folder):
        os.mkdir(folder)

def extract_keyframe(videopath , target_frames):
    cap = cv2.VideoCapture(videopath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #影片偵數小於目標偵數直接回傳
    if length < target_frames:
        return []
    dif_list=[]
    pre_frame = None
    cur_frame = None
    index = 0

    #先讀第一張，避免後面迴圈執行太多次if判斷
    ret, pre_frame = cap.read()
    if not ret:
        print("讀取frame時出現錯誤")
        return

    while (True):
        ret, cur_frame = cap.read()
        if not ret:
            print("讀取frame時出現錯誤")
            break
        dif = cv2.absdiff(cur_frame, pre_frame)
        dif_sum = np.sum(dif)
        dif_sum_mean = dif_sum / (dif.shape[0] * dif.shape[1])  # dif.shape[0] * dif.shape[1] : 影格的總像素個數
        dif_list.append([index , dif_sum_mean ,[pre_frame,index] ,[cur_frame,index+1] ])
        index+=1
        pre_frame = cur_frame
    cap.release()

    # 根據影格間的差分進行排序
    dif_list.sort(key=lambda f: f[1] , reverse=True)

    frame_list = []     #選取到的影格(含影格的索引)   [ img , index ]
    img_list = []       #選取到的影格(去除影格的索引) [ img ]
    select_frame_index = []  #統計選到影格的索引，避免選到重覆的frame

    i=0
    while(True):
        frame = dif_list[i][2]
        index = frame[1]
        if not (index in select_frame_index):
            frame_list.append(frame)
            select_frame_index.append(index)

        frame = dif_list[i][3]
        index = frame[1]
        if not (index in select_frame_index):
            frame_list.append(frame)
            select_frame_index.append(index)

        if len(frame_list) >= target_frames:
            break
        i+=1
    frame_list = frame_list[0:target_frames]
    frame_list.sort(key=lambda f: f[1])           #按照id排序
    for i in range(target_frames):
        img_list.append(frame_list[i][0])

    return img_list

video_to_frame("DFDC_FAKE")