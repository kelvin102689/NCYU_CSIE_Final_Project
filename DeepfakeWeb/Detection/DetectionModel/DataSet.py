from torch.utils.data.dataset import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import glob
import imutils
import torch
import cv2
import numpy
import uuid
import os
import sys


from django.shortcuts import render
from django.http.response import HttpResponse


Frame_Datum = 60
Frame_Width = 180
Frame_Height = 180

#transforms資料增強 (含資料標準化)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
Transforms = transforms.Compose( [
    #用 io.imread or cv2讀取到的圖片格式為ndarray
    #而transforms在做預處理時 格式要求為PILImage
    transforms.ToPILImage(),
    transforms.Resize((Frame_Width , Frame_Height)),
    #模型在處理資料時大多用tensor表示 , 故需再做一次轉換
    transforms.ToTensor(),
    transforms.Normalize(mean , std)
] )

class VideoSet(Dataset):
    def __init__(self, videos_frames, transform=Transforms):
        self.videos_frames = videos_frames            #Videos存的是個別影片的總偵
        self.transform = transform
        self.FrameCount = Frame_Datum   # 一部影片要的偵數

    def __len__(self):
        return len(self.videos_frames)

    #返回單一影片處理過後的frames
    def __getitem__(self, idx):
        frames = self.videos_frames[idx]
        frames = self.face_detect(frames)
        frames = self.extract_keyframe(frames)
        frames = self.get_model_input(frames)
        return frames

    def face_detect(self, frames):
        fuck = 0
        # 偵測人臉
        crop_imgs = []
        mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda')
        # print(len(frames))
        for img in frames:
            fuck+=1
            # 縮小圖片
            img = imutils.resize(img, height=600, width=600)

            # OpenCV預設讀取channel為BGR  PIL為RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 將 NumPy 陣列轉換為 PIL 影象
            img = Image.fromarray(img)

            boxes, probs = mtcnn.detect(img)
            try:
                for box, prob in zip(boxes, probs):
                    # print(len(boxes))
                    # print(box)
                    #print("Face Detect Precision : %f" % (prob))
                    # 偵測到的box人臉機率大於0.6才進行儲存
                    if prob > 0.8:
                        for i in range(0, 4):  # box[0],box[1]左上角的x,y座標  box[2],box[3]右下角x,y座標
                            if i == 0 or i == 1:
                                box[i] -= 5
                            else:
                                box[i] += 5
                        try:
                            # filename = str(uuid.uuid4().hex)
                            Path = str(fuck) + '.jpg'
                            img = img.crop(box)
                            img = img.resize((600, 600))
                            img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
                            #img.save(Path)
                            #img = cv2.imread(Path)  # 可再優化(直接將img轉cv2格式)
                            # os.remove(Path)
                            crop_imgs.append(img)
                        except Exception as e:
                            print(e)
                            print("儲存或讀取人像圖片時發生錯誤!!")
                    else:
                        print("Face Detect Precision is too low ---> abort!")
            except Exception as e:
                print("Detect face fail !")
        print('偵測到的人臉偵數 : %d張' %(len(crop_imgs)))
        return crop_imgs

    def extract_keyframe(self, frames):

        dif_list = []
        for index in range(0, len(frames) - 1):
            pre_frame = frames[index]
            cur_frame = frames[index + 1]

            # 算差分
            dif = cv2.absdiff(cur_frame, pre_frame)
            dif_sum = np.sum(dif)
            dif_sum_mean = dif_sum / (dif.shape[0] * dif.shape[1])  # dif.shape[0] * dif.shape[1] : 影格的總像素個數
            dif_list.append([index, dif_sum_mean, [pre_frame, index], [cur_frame, index + 1]])

        # 根據影格間的差分進行排序
        dif_list.sort(key=lambda f: f[1], reverse=True)

        frame_list = []  # 選取到的影格(含影格的索引)   [ img , index ]
        img_list = []  # 選取到的影格(去除影格的索引) [ img ]
        select_frame_index = []  # 統計選到影格的索引，避免選到重覆的frame

        for index in range(0, len(dif_list)):
            frame = dif_list[index][2]
            index = frame[1]
            if not (index in select_frame_index):
                frame_list.append(frame)
                select_frame_index.append(index)

            frame = dif_list[index][3]
            index = frame[1]
            if not (index in select_frame_index):
                frame_list.append(frame)
                select_frame_index.append(index)

            if len(frame_list) >= Frame_Datum:
                break

        frame_list = frame_list[0:Frame_Datum]
        frame_list.sort(key=lambda f: f[1])  # 按照id排序
        for i in range(Frame_Datum):
            img_list.append(frame_list[i][0])

        return img_list

    def get_model_input(self, frames, Transforms=Transforms):
        for index in range(0, len(frames)):
            frames[index] = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)  # 將BGR轉成RGB(PIL是RGB格式)
            frames[index] = Transforms(frames[index])
        # 取要的偵數
        frames = frames[:Frame_Datum]
        frames = torch.stack(frames)
        return frames








