from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
import torch
import cv2

Frame_Datum =  36#原本85
Frame_Width = 180 #原本110
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

class Video_Dataset(Dataset):
    #labels - > csv檔
    def __init__(self, video_type, labels, transform=Transforms):
        self.video_type = video_type
        self.labels = labels
        self.transform = transform
        # 一部影片要的偵數
        self.FrameCount = Frame_Datum

    def __len__(self):
        return len(self.video_type)

    #須返回單一個影片的frames and label
    def __getitem__(self, idx):
        video_path = self.video_type[idx]
        frames = self.extract_frame(video_path)
        #取要的偵數
        frames = frames[:self.FrameCount]
        frames = torch.stack(frames)
        VideoPrefix = Get_VideoPrefix(video_path)
        index = self.labels.index[self.labels["VideoName"]==(VideoPrefix)].tolist()[0]
        name  =  self.labels.at[index , 'VideoName']
        lable = self.labels.at[index , 'label']
        if lable=='REAL':
            lable = 1
        else:
            lable = 0
        #print(name)
        #print(lable)
        return frames , lable


    def extract_frame(self , video_path):
        VideoName = (video_path.split("\\"))[1]
        frames = []
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                #print("extracting %s  frame was done" %(VideoName))
                break
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)     #將BGR轉成RGB(PIL是RGB格式)
            frame = self.transform(frame)
            frames.append(frame)
        return frames

#讀取資料
def GetVideo(Path, Label):
    TempSet = []
    VideoType = []    #儲存影片對應的種類 Real : 1, Fake : 0
    for path in Path:
        TempSet += (glob.glob(path))
    VideoSet = []

    for video in TempSet :
        if IsValidVideo(video, Label):
            Type = GetVideoType(video, Label)
            if Type == 'REAL':
                VideoType.append(1)
            else:
                VideoType.append(0)
            VideoSet.append(video)
    print("Total videos count : %d" %(len(VideoSet)))
    return VideoSet, VideoType

#驗證影片是否滿足要求，且有記錄在csv檔
def IsValidVideo(video, Label):
    cap = cv2.VideoCapture(video)
    Frame_Count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 拿取影片的前綴名
    VideoPrefix = Get_VideoPrefix(video)
    target = Label["VideoName"].isin([VideoPrefix])
    if Frame_Count < Frame_Datum or Label[target].empty:
        return False
    return True

def Number_Of_Fake_And_Real(videoset, Label):
    Real = 0
    Fake = 0
    for video_path in videoset:
        Type = GetVideoType(video_path, Label)
        if Type == 'REAL':
            Real+=1
        elif Type == 'FAKE':
            Fake+=1
    return(Real , Fake)

#拿取影片的前綴名
def Get_VideoPrefix(video):
    VideoName = video.split('\\')[1]
    if "_original0" in VideoName:
        VideoName = VideoName.replace("_original0", "")
    elif "_contrast0" in VideoName:
        VideoName = VideoName.replace("_contrast0", "")
    elif "_color_temperature0" in VideoName:
        VideoName = VideoName.replace("_color_temperature0", "")
    return VideoName

def GetVideoType(video_path, Label):
    # 拿取影片的前綴名
    VideoPrefix = Get_VideoPrefix(video_path)
    index = Label.index[Label["VideoName"] == (VideoPrefix)].tolist()[0]
    # name = self.labels.at[index, 'VideoName']
    Type = Label.at[index, 'label']
    return Type

#將原影片名稱和其對應的預處理影片用dictionary建立好關係
def Build_VideoMap(VideoSet, VideoType):
    VideoList = []
    VideoLabel = []
    video_map = {}
    for index in range(0, len(VideoSet)):
        video = VideoSet[index]
        videoname = Get_VideoPrefix(video)
        if videoname in video_map:
            video_map[videoname].append(video)
        else:
            video_map[videoname] = [video]
            VideoList.append(videoname)
            VideoLabel.append(VideoType[index])
    return VideoList, VideoLabel, video_map