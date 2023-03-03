from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def Plot_FinalResult(Tloss, VLoss, TAccuracy, VAccuracy, fold_count):
    fig, ax = plt.subplots(1, 2)
    plt.figure(figsize=(10, 7))
    folds = range(1, fold_count + 1)
    ax[0].set_title('Loss')
    ax[0].plot(folds, Tloss, markersize="16", marker=".", color='red', label="Train Loss")
    ax[0].plot(folds, VLoss, markersize="16", marker=".", color='blue', label="Validate Loss")
    ax[0].set(xlabel="Folds", ylabel="Minimun Loss")
    ax[0].legend(loc='best')

    ax[1].set_title('Accuracy')
    ax[1].plot(folds, TAccuracy, markersize="16", marker=".", color='red', label="Train Accuracy")
    ax[1].plot(folds, VAccuracy, markersize="16", marker=".", color='blue', label="Validate Accuracy")
    ax[1].set(xlabel="Folds", ylabel="Maximum Accuracy")
    ax[1].legend(loc='best')
    fig.tight_layout()
    fig.savefig('Experiments/Result.jpg')


def Plot_loss(loss, epoch_num, path, fold_num):
    plt.figure(figsize=(7, 7))
    epochs = range(1,epoch_num+1)
    train_avg_loss = loss[0]
    #train_avg_acc  = data[1]
    valid_avg_loss = loss[1]
    #valid_avg_acc  = data[3]
    plt.title('Fold'+str(fold_num))
    plt.plot(epochs, train_avg_loss, markersize="16", marker=".", color='red', label = "Training Loss")
    plt.plot(epochs, valid_avg_loss, markersize="16", marker=".", color='blue', label="Validate Loss")
    plt.xlabel('Epochs')  # 設定 x 軸標題
    plt.ylabel('Average Loss')  # 設定 y 軸標題
    plt.legend(loc='best')
    plt.savefig(path+'/Loss.jpg')

def Plot_accuracy(accuracy, epoch_num, path, fold_num):
    plt.figure(figsize=(7, 7))
    epochs = range(1,epoch_num+1)
    train_avg_acc = accuracy[0]
    valid_avg_acc = accuracy[1]
    plt.title('Fold' + str(fold_num))
    plt.plot(epochs, train_avg_acc, markersize="16", marker=".", color='red', label = "Training Accuracy")
    plt.plot(epochs, valid_avg_acc, markersize="16", marker=".", color='blue', label="Validate Accuracy")
    plt.xlabel('Epochs')  # 設定 x 軸標題
    plt.ylabel('Average Accuracy')  # 設定 y 軸標題
    plt.legend(loc='best')
    plt.savefig(path + '/Accuracy.jpg')


def Plot_TrainingInfo(loss, accuracy, epoch_num, path, fold_num):
    Plot_loss(loss, epoch_num, path, fold_num)
    Plot_accuracy(accuracy, epoch_num, path, fold_num)
    #plt.show()

def Plot_DataPie(TrainDataCount, ValidateDataCount, path):
    explode = (0.05, 0.05)
    dfA = pd.DataFrame([
        ['Real', TrainDataCount[0]], ['Fake', TrainDataCount[1]]],
        columns=['Type', 'Count'])
    dfB = pd.DataFrame([
        ['Real', ValidateDataCount[0]], ['Fake', ValidateDataCount[1]]],
        columns=['Type', 'Count'])
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Video Type Count')
    ax[0].set_title('Training DataSet')
    ax[0].pie(dfA['Count'], labels=dfA['Type'], autopct=lambda pct: autopct_format(pct, dfA['Count']), shadow=True,
              explode=explode)
    ax[1].set_title('Validate DataSet')
    ax[1].pie(dfB['Count'], labels=dfB['Type'], autopct=lambda pct: autopct_format(pct, dfB['Count']), shadow=True,
              explode=explode)
    fig.savefig(path+'/VideoTypeRatio.jpg')

def Plot_Testing_DataPie(TestingDataCount):
    explode = (0.05, 0.05)
    dfA = pd.DataFrame([
        ['Real', TestingDataCount[0]], ['Fake', TestingDataCount[1]]],
        columns=['Type', 'Count'])
    plt.title('Testing DataSet')
    plt.pie(dfA['Count'], labels=dfA['Type'], autopct = lambda pct: autopct_format(pct, dfA['Count']), shadow = True, explode=explode)
    plt.savefig('Experiments/Test_DataPie.jpg')

#自訂義圓餅圖顯現資料的格式
def autopct_format(pct, values):
    total = sum(values)
    val = int(round(pct * total / 100.0))
    return '{:1.2f}%\n({v:d})'.format(pct, v=val)

#畫出模型跑完n個fold train和valid的統計數據(平均值)
def Plot_ResultTable(Train_Average_Loss, Train_Average_Acc, Validate_Average_Loss, Validate_Average_Accuracy):
    plt.figure(figsize=(5, 3), facecolor='#f5ebeb')
    column_labels = ["Average_Loss", "Average_Acc"]
    row_labels = ["Train", "Validate"]
    data = [[round(Train_Average_Loss, 2), round(Train_Average_Acc, 2)],
            [round(Validate_Average_Loss, 2), round(Validate_Average_Accuracy, 2)]]
    plt.axis('square')
    plt.axis('off')
    plt.title('Result Table')
    table = plt.table(cellText=data, colLabels=column_labels, rowLabels=row_labels,loc="center",
              rowColours = ["lightblue"] * 2, colColours = ["lightblue"] * 2)
    table.scale(1, 2)
    plt.savefig("Experiments/ResultTable.jpg")

def Plot_Confusion_Matrix(matrix, path):
    for index in range(0, len(matrix)):
        cm = matrix[index]
        classes = ['FAKE', 'REAL']
        fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=classes, colorbar=True)
        plt.savefig(path+"/Confusion_Matrix_" + str(index) + ".jpg", bbox_inches = 'tight')


#Plot_ResultTable(0.523, 85.458974, 0.785569, 77.22587)

#epoch_num = range(1,6)
'''
test = [20,33,55.9,66,87.2]
valid = [11,24,60,77.9,84.6]
test2 = [23,33,5.9,100,87.2]
valid2 = [11,16,60,70.9,84.6]
Plot_FinalResult(test, valid, test2, valid2, 5)
'''
#data = []
#data.append(test)
#data.append(valid)
#Plot_img(data, epoch_num)
#test=[]
#test.append(99)
#test.append(88)
#Plot_Testing_DataPie(test)

'''
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
y_pred = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1]
cm = confusion_matrix(y_true, y_pred)  # reverse true/pred and label values
'''