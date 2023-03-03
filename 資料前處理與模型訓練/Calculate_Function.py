#取回預測和實際的數值 -> 以0和1表示
def Get_Predict_And_Actual_Info(outputs, targets):
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    targets.view(1, -1)
    return pred.tolist()[0], targets.tolist()



def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    # targets.view(1, -1) 將tensor reshape成[1 * n] n由targets的維度決定 , 例如一次拿4個影片訓練 target的長度就是4
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size


def F1Score_Based_on_Fake(tn, fn, fp):
    Precision = tn / (tn + fn)
    Recall = tn / (tn + fp)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    return Precision, Recall, F1

def F1Score_Based_on_Real(tp, fp, fn):
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    return Precision, Recall, F1