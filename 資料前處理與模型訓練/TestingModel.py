import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from DataSet import  *
from Model import  *
from AverageMeter import  *
from Ploting import *
from Calculate_Function import *

def TestingModel():
    Model_File_Path = "model.pt"
    Title = ["VideoName", "label"]
    Label = pd.read_csv('Video_Label/deepfake_metadata_1114.csv', names=Title)
    Label = Label.dropna()
    Testing_Video_Path= ["test_inc/*.mp4"]

    Test_VideoSet, _ = GetVideo(Testing_Video_Path, Label)
    Test_VideoSet = Get_Original_Video(Test_VideoSet)
    #Test_VideoSet = Get_Real_Video(Test_VideoSet, Label)
    RealNum, FakeNum = Number_Of_Fake_And_Real(Test_VideoSet, Label)
    print("Test_VideoSet : Total", len(Test_VideoSet), "videos" + " -----> REAL :", RealNum, " FAKE :", FakeNum, )

    # 調整資料集真假影片的比例-> 1 : 1
    #Test_VideoSet = Adjust_Ratio(RealNum, FakeNum, Test_VideoSet, Label)
    #RealNum, FakeNum = Number_Of_Fake_And_Real(Test_VideoSet, Label)
    #print("Test_VideoSet : Total", len(Test_VideoSet), "videos" + " -----> REAL :", RealNum, " FAKE :", FakeNum, )
    Testing_Video_type_count = [RealNum, FakeNum]

    Testing_Dataset = Video_Dataset(Test_VideoSet, Label)
    Testing_loader = DataLoader(Testing_Dataset, batch_size=1, shuffle=True, num_workers=0)
    model = Model(classes=2).cuda()
    try:
        #model.eval()
        model.load_state_dict(torch.load(Model_File_Path))
        #Testing完回傳confusion matrix info
        cm = StartTesting(model, Testing_loader)
        Plot_Testing_DataPie(Testing_Video_type_count)
        Plot_Confusion_Matrix([cm], "Experiments")
    except FileNotFoundError:
        print("The model not find !!")

#Testing完回傳confusion matrix info
def StartTesting(model, Testing_loader):
    model.eval()  # Optional when not using Model Specific layer
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    accuracies = AverageMeter()
    predicted = []
    actual = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(Testing_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
            _, Result = model(inputs)
            loss = torch.mean(criterion(Result, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(Result, targets.type(torch.cuda.LongTensor))

            pre, act = Get_Predict_And_Actual_Info(Result, targets.type(torch.cuda.LongTensor))
            predicted.extend(pre)
            actual.extend(act)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            print("Batch[%4d / %4d] ------>(Testing) Loss : %f, Accuracy : %f"
                  % ( i + 1, len(Testing_loader), losses.val, accuracies.val))
    print("[Testing] Average Loss : %f, Average Accuracy : %f" %(losses.avg, accuracies.avg))

    cm = confusion_matrix(actual, predicted)
    tn, fp, fn, tp = cm.ravel()

    # Based on Fake (用假影片當作正樣本)
    Precision, Recall, F1 = F1Score_Based_on_Fake(tn, fn, fp)
    print("[Testing] (Score based on fake) Precision : %f,  Recall : %f,  F1 : %f" % (Precision, Recall, F1))
    # Based on Real (用真影片當作正樣本)
    Precision, Recall, F1 = F1Score_Based_on_Real(tp, fp, fn)
    print("[Testing] (Score based on real) Precision : %f,  Recall : %f,  F1 : %f" % (Precision, Recall, F1))

    return cm





#調整資料集真假影片的比例-> 1 : 1
def Adjust_Ratio(Real_Count, Fake_Count, VideoSet, Label):
    if Real_Count < Fake_Count:
        Fake_Count = Real_Count
    else:
        Real_Count = Fake_Count
    random.shuffle(VideoSet)
    Final_DataSet = []
    for video in VideoSet:
        if Real_Count==0 and Fake_Count==0:
            break
        VideoPrefix = Get_VideoPrefix(video)
        index = Label.index[Label["VideoName"] == (VideoPrefix)].tolist()[0]
        Type = Label.at[index, 'label']
        if Type == 'FAKE' and Fake_Count > 0:
            Final_DataSet.append(video)
            Fake_Count-=1
        elif Type == 'REAL' and Real_Count > 0:
            Final_DataSet.append(video)
            Real_Count-=1

    random.shuffle(Final_DataSet)
    return Final_DataSet

def Get_Original_Video(videoset):
    Original = []
    for videoname in videoset:
        if not("_contrast0" in videoname) and not("_color_temperature0" in videoname):
            Original.append(videoname)
    return Original

def Get_Real_Video(videoset, Label):
    Real = []
    for videoname in videoset:
        type = GetVideoType(videoname, Label)
        if type == "REAL":
            Real.append(videoname)
    return Real

def Get_Fake_Video(videoset, Label):
    Fake = []
    for videoname in videoset:
        type = GetVideoType(videoname, Label)
        if type == "FAKE":
            Fake.append(videoname)
    return Fake

#TestingModel()