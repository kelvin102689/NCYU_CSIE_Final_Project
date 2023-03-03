from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import random
import pandas as pd
import numpy as np
import os
from TestingModel import *
from DataSet import *
from Model import *
from Ploting import *
from AverageMeter import  *
from Calculate_Function import *


print(torch.cuda.is_available())
print( torch.cuda.device_count())
Model_File_Path = "model.pt"
VideoSet = []
Train_VideoSet = []
Valid_VideoSet = []
Label = []
#Train_Ratio = 0.8
#Valid_Ratio = 0.2

'''
#打散資料並按比例分成訓練，驗證的資料集
def DataShuffle(VideoSet):
    random.shuffle(VideoSet)
    Train_VideoSet = VideoSet[: int(len(VideoSet)*Train_Ratio) ]
    Valid_VideoSet = VideoSet[int(len(VideoSet)*Train_Ratio):]

    print("Total train videos count : %d" %(len(Train_VideoSet)))
    print("Total valid videos count : %d" %(len(Valid_VideoSet)))
    return(Train_VideoSet , Valid_VideoSet)
'''

def TrainEpoch(epoch, num_epochs, Train_loader, model, criterion, optimizer):
    # train mode
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (inputs, targets) in enumerate(Train_loader):
        #torch.cuda.empty_cache()
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor)
            inputs = inputs.cuda()
        _,Result = model(inputs)
        loss = criterion(Result, targets.type(torch.cuda.LongTensor))
        acc = calculate_accuracy(Result, targets.type(torch.cuda.LongTensor))
        #The item() method extracts the loss’s value as a Python float.
        #print(loss.item())
        #print(inputs.size(0))
        losses.update(loss.item(), 1)
        accuracies.update(acc, 1)
        optimizer.zero_grad()            #需先清空 否則會受上一個batch grad影響
        loss.backward()                  #反向傳播計算當前batch 的 grad
        optimizer.step()                 #根據反向梯度更新參數

        print("Epoch : [%2d / %2d] Batch[%4d / %4d] ------>(Train) Loss : %f, Accuracy : %f"
             %(epoch, num_epochs, i+1, len(Train_loader), losses.val, accuracies.val))

    #這邊要改成平均loss**************
    return losses.avg , accuracies.avg



def ValidEpoch(epoch, num_epochs, Validate_loader, model ,criterion):
    model.eval()  # Optional when not using Model Specific layer
    losses = AverageMeter()
    accuracies = AverageMeter()

    predicted = []
    actual = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(Validate_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
            _,Result = model(inputs)
            loss = torch.mean(criterion(Result, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(Result,targets.type(torch.cuda.LongTensor))

            pre, act = Get_Predict_And_Actual_Info(Result,targets.type(torch.cuda.LongTensor))
            predicted.extend(pre)
            actual.extend(act)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            print("Epoch : [%2d / %2d] Batch[%4d / %4d] ------>(Validate) Loss : %f, Accuracy : %f"
                 % (epoch, num_epochs, i+1, len(Validate_loader), losses.val, accuracies.val))

    #print(predicted)
    #print(actual)
    #tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    cm = confusion_matrix(actual, predicted)
    #print(cm)
    #print(confusion_matrix_info)
    #print(predicted)
    #print(actual)
    return losses.avg, accuracies.avg, cm

def TrainingModel(model, Train_loader, Validate_loader, fold_num):

    global min_valid_loss  # 用來當作儲存表現最好模型的依據
    lr = 1e-5                # learning rate
    num_epochs = 3          # number of epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    # store confusion matrix
    matrix = []
    train_avg_loss = []
    train_accuracy = []
    valid_avg_loss = []
    valid_accuracy = []
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = TrainEpoch(epoch, num_epochs, Train_loader, model, criterion, optimizer)
        validate_loss, validate_acc, cm = ValidEpoch(epoch, num_epochs, Validate_loader, model, criterion)
        matrix.append(cm)
        train_avg_loss.append(train_loss)
        valid_avg_loss.append(validate_loss)
        train_accuracy.append(train_acc)
        valid_accuracy.append(validate_acc)
        print("Epoch : [%2d / %2d]  ------>(Train) Average Loss : %f, Average Accuracy : %f"
              % (epoch, num_epochs, train_loss, train_acc))
        print("Epoch : [%2d / %2d]  ------>(Validate) Average Loss : %f, Average Accuracy : %f"
              % (epoch, num_epochs, validate_loss, validate_acc))

        tn, fp, fn, tp = cm.ravel()

        # Based on Fake (用假影片當作正樣本)
        Precision, Recall, F1 = F1Score_Based_on_Fake(tn, fn, fp)
        print("Epoch : [%2d / %2d]  ------>(Score based on fake) Precision : %f,  Recall : %f,  F1 : %f"
              % (epoch, num_epochs, Precision, Recall, F1))

        # Based on Real (用真影片當作正樣本)
        Precision, Recall, F1 = F1Score_Based_on_Real(tp, fp, fn)
        print("Epoch : [%2d / %2d]  ------>(Score based on real) Precision : %f,  Recall : %f,  F1 : %f"
              % (epoch, num_epochs, Precision, Recall, F1))

        # 如果模型在驗證時表現比以往好 則儲存其參數
        if min_valid_loss > validate_loss:
            print("Validation Loss Decreased(%f ----> %f)\tSaving The Model"%(min_valid_loss, validate_loss))
            torch.save(model.state_dict(), Model_File_Path)
            min_valid_loss = validate_loss
    Loss = []
    Accuracy = []
    Loss.append(train_avg_loss)
    Loss.append(valid_avg_loss)
    Accuracy.append(train_accuracy)
    Accuracy.append(valid_accuracy)
    #畫loss , accurate折線圖
    Plot_TrainingInfo(loss=Loss, accuracy=Accuracy, epoch_num=num_epochs, path="Experiments/Fold"+str(fold_num), fold_num=fold_num)
    return Loss, Accuracy, matrix

def Train_With_Stratifiedkfold(VideoSet, VideoType , VideoMap, Label):
    split = 3
    count = 1
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    Fold_Loss = []      #紀錄每一個Fold訓練完的train and validate Loss
    Fold_Acc = []       #紀錄每一個Fold訓練完的train and validate Acc
    Fold_train_max_acc = []
    Fold_valid_max_acc = []
    Fold_train_min_loss = []
    Fold_valid_min_loss = []

    for train_index, valid_index in skf.split(VideoSet, VideoType):
        if not os.path.isdir("Experiments/Fold"+str(count)):
            os.mkdir("Experiments/Fold"+str(count))
        Train_VideoSet = []
        Valid_VideoSet = []
        for index in train_index:
            video = VideoMap[VideoSet[index]]
            Train_VideoSet.extend(video)
        for index in valid_index:
            video = VideoMap[VideoSet[index]]
            Valid_VideoSet.extend(video)
        random.shuffle(Train_VideoSet)
        random.shuffle(Valid_VideoSet)
        RealNum, FakeNum = Number_Of_Fake_And_Real(Train_VideoSet, Label)
        print("Fold [", count,"/", str(split), "] Train_VideoSet : Total", len(Train_VideoSet), "videos" + " -----> REAL :", RealNum,
              " FAKE :", FakeNum, )
        Train_Video_type_count = [RealNum, FakeNum]
        RealNum, FakeNum = Number_Of_Fake_And_Real(Valid_VideoSet, Label)
        print("Fold [", count,"/", str(split), "] Valid_VideoSet : Total", len(Valid_VideoSet), "videos" + " -----> REAL :", RealNum,
              " FAKE :", FakeNum, )
        Validate_Video_type_count = [RealNum, FakeNum]

        #畫資料的比例圖
        Plot_DataPie(Train_Video_type_count, Validate_Video_type_count, "Experiments/Fold"+str(count))

        Training_Dataset = Video_Dataset(Train_VideoSet, Label)
        Validate_Dataset = Video_Dataset(Valid_VideoSet, Label)
        Train_loader = DataLoader(Training_Dataset, batch_size=2, shuffle=True, num_workers=0)  # num_workers代表要用多少個subprocess載入data
        Validate_loader = DataLoader(Validate_Dataset, batch_size=2, shuffle=True, num_workers=0)

        model = Model(classes=2).cuda()

        print("Fold [", count,"/", str(split), "] start training .....")
        Loss, Acc, matrix = TrainingModel(model, Train_loader, Validate_loader, count)
        Fold_Loss.append(Loss)
        Fold_Acc.append(Acc)
        Plot_Confusion_Matrix(matrix, "Experiments/Fold"+str(count))

        count+=1

    # Fold_Acc[i][0] 代表第i個fold的train accuracy , Fold_Acc[i][1] 代表第i個fold的valid accuracy 以此類推
    for i in range(0,count-1):
        Fold_train_max_acc.append(max(Fold_Acc[i][0]))
        Fold_valid_max_acc.append(max(Fold_Acc[i][1]))
        Fold_train_min_loss.append(min(Fold_Loss[i][0]))
        Fold_valid_min_loss.append(min(Fold_Loss[i][1]))
        print("Fold [", i+1,"/", str(split), "] (Train) max accuracy :",Fold_train_max_acc[i],
              ",  min loss :",Fold_train_min_loss[i],)
        print("Fold [", i+1,"/", str(split), "] (Validate) max accuracy :", Fold_valid_max_acc[i],
              ",  min loss :", Fold_valid_min_loss[i], )

    Train_Average_Loss = sum(Fold_train_min_loss) / len(Fold_train_min_loss)
    Train_Average_Acc  = sum(Fold_train_max_acc) / len(Fold_train_max_acc)
    Validate_Average_Loss = sum(Fold_valid_min_loss) / len(Fold_valid_min_loss)
    Validate_Average_Accuracy = sum(Fold_valid_max_acc) / len(Fold_valid_max_acc)
    print("Final Result : ")
    print("Train Average Loss :", Train_Average_Loss, )
    print("Train Average Accuracy :", Train_Average_Acc, )
    print("Validate Average Loss :", Validate_Average_Loss, )
    print("Validate Average Accuracy :",Validate_Average_Accuracy,)
    Plot_ResultTable(Train_Average_Loss, Train_Average_Acc, Validate_Average_Loss, Validate_Average_Accuracy)
    Plot_FinalResult(Fold_train_min_loss, Fold_valid_min_loss, Fold_train_max_acc, Fold_valid_max_acc, count-1)


if not os.path.isdir("Experiments"):
    os.mkdir("Experiments")

min_valid_loss = np.inf  # 用來當作儲存表現最好模型的依據

Title = ["VideoName","label"]
Label = pd.read_csv('Training_Label/deepfake_metadata_0704.csv' , names = Title)
Label = Label.dropna()

Training_Video_Path = ["DFDC_FAKE/*.mp4"]
VideoSet, VideoType= GetVideo(Training_Video_Path, Label)

#取只含前綴的videoset 對應的videotype和對應的map
VideoSet, VideoType, VideoMap = Build_VideoMap(VideoSet, VideoType)

#用Stratified kfold做訓練並交叉驗證
Train_With_Stratifiedkfold(VideoSet, VideoType, VideoMap ,Label)





#----------test--------------
'''
Training_Dataset = Video_Dataset(Train_VideoSet , Label)
Validate_Dataset = Video_Dataset(Valid_VideoSet , Label)
Train_loader = DataLoader(Training_Dataset , batch_size = 4 ,shuffle = True, num_workers = 0)     #num_workers代表要用多少個subprocess載入data
Validate_loader = DataLoader(Validate_Dataset , batch_size = 4 ,shuffle = True, num_workers = 0)

model = Model(classes=2).cuda()
try:
    model.load_state_dict(torch.load(Model_File_Path))
except FileNotFoundError:
    pass
'''
#TrainingModel(model, Train_loader, Validate_loader)
#TestingModel()
#Plot_DataPie(Train_Video_type_count, Validate_Video_type_count)