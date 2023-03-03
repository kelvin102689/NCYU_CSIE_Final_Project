from django.shortcuts import render
from django.http.response import HttpResponse
from django.conf import settings
from django.contrib import messages
from .forms import  UploadVideoForm
from .models import Video
from .DetectionModel.Model import *
#from .DetectionModel.video_preprocess import *
from .DetectionModel.DataSet import *
from torch.utils.data import DataLoader
from datetime import datetime, date
import cv2
import os
from django.http import JsonResponse

# Create your views here.
model_loc = "C:\\Users\\user\\Desktop\\Deepfake_Web\\DeepfakeWeb\\Detection\\DetectionModel\\model.pt"
model = Model(classes=2).cuda()
model.load_state_dict(torch.load(model_loc))

def UploadVideo(request):
    Context = {}
    Context['form'] = ''
    Context['video'] = ''
    Form = UploadVideoForm()
    if request.POST:
        Form = UploadVideoForm(request.POST, request.FILES)
        if Form.is_valid():
            if Form.check_type_and_size() == 'Valid':
                UploadVideo = Form.save(commit=False)
                UploadVideo.name = request.FILES['video'].name
                UploadVideo.save()
                Context['video'] = UploadVideo
            # 檔案型別或大小不符合
            else:
                messages.error(request, "檔案型別或大小不符合 !   (型別需為.mp4檔 大小請小於50MB)")

    Context['form'] = Form
    return render(request, "Detection/UploadFiletest.html", Context)



def Detection(request):
    CleanVideoData()         #清除創建超過兩天的影片物件
    if request.POST:
        VideoID = request.POST.get('VideoID')
        video = Video.objects.get(id = VideoID)
        if video != None:
            try:
                Type, FakeProb = Detect(video)
                Prob = str(round(FakeProb*100, 2))+'%'
                Data = {'Type' : Type, 'Prob' : Prob}
                return JsonResponse(Data)
            except Exception as e:
                print("偵測發生錯誤")
                print(e)
        else:
            print("Video does not find in database !")
        '''
        if  Form.is_valid():
            if Form.check_type_and_size() == 'Valid':
                UploadVideo = Form.save(commit=False)
                UploadVideo.name = request.FILES['video'].name
                UploadVideo.save()
                try:
                    video = Video.objects.last()
                    Context['video'] = video

                    Type, Prob = Detect(video)
                    Context['Type'] = Type
                    Context['Prob'] = str(round(Prob*100, 2))+'%'
                    print(Type)
                    print(str(round(Prob*100, 2)))

                    print(video.create_time)
                except Exception as e:
                    print(e)
            # 檔案型別或大小不符合
            else :
                messages.error(request, "檔案型別或大小不符合 !   (型別需為.mp4檔 大小請小於50MB)")

    Context['form'] = Form
    '''
    #return render(request, "Detection/UploadFiletest.html", Context)
    #return HttpResponse("My First Django App.")



# has problem
def CleanVideoData():
    records = Video.objects.all()
    for record in records[:]:            #迴圈跑的是records的copy
        try :
            if created_time_is_more_than_twodays(record.create_time):
                FileName = record.video.url.split('/')[-1]
                os.remove(os.path.join(settings.MEDIA_ROOT) + '\\videos\\' + FileName)
                records.delete()
        except Exception as e:
            print(e)
            pass

def created_time_is_more_than_twodays(CreatedTime):
    CurrentTime = datetime.now()
    dA = date(CurrentTime.year, CurrentTime.month, CurrentTime.day)
    dB = date(CreatedTime.year, CreatedTime.month, CreatedTime.day)

    diff = dA - dB
    return diff.days >= 2


def Detect(video):
    #os.path.join(settings.MEDIA_ROOT) + '\\videos\\' + video.name
    frames = []
    FileName = video.video.url.split('/')[-1]

    cap = cv2.VideoCapture(os.path.join(settings.MEDIA_ROOT) + '\\videos\\' + FileName)
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print("影片總偵數 : %d張" %(len(frames)))
    Type = ''
    Probability = 0
    try:
        videos_frames = [frames]
        ValidationSet = VideoSet(videos_frames)
        validatiob_loader = DataLoader(ValidationSet, batch_size=1)
        #model_loc = "C:\\Users\\user\\Desktop\\Deepfake_Web\\DeepfakeWeb\\Detection\\DetectionModel\\model2.pt"
        #model = Model(classes=2).cuda()

        #print(model.load_state_dict(torch.load(model_loc)))


        model.eval()  # Optional when not using Model Specific layer
        with torch.no_grad():
            for i, inputs in enumerate(validatiob_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    _, result = model(inputs)
                    #print(result)
                    m = nn.Softmax(dim=1)
                    result = m(result)
                    #print(result[0][1])
                    _, pred = result.topk(1, 1, True)
                    pred = pred.t()

                    predict = pred.item()
                    FakeProb = result[0][0].item()             #影片被造假的機率
                    if(predict == 0):
                        Type = '0'
                    else:
                        Type = '1'
                else:
                    print("Cuda is not available !")

    except FileNotFoundError as e:
        print(e)
        print("模型偵測失敗")
    print("判別完成 --> " + Type + ", " + str(Probability))
    return Type, FakeProb



