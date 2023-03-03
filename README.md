# NCYU_CSIE_Final_Project
112嘉大資工畢業專題 ( 基於卷積神經網絡和長短期記憶模型實作換臉影片偵測 )

# Python版本 
大於3.10

# 訓練用資料集
1. DFDC
2. Celeb-DFV2
3. FF++

# 資料處理與模型訓練
程式碼請參考 `資料前處理與模型訓練`

**資料集前處理流程**  
KeyFrame選取 -> MTCNN擷取人臉 -> 對比、色溫等資料增強

# 訓練資料Label
請參考 `Video_Label`

# 使用者介面
以Django為框架進行架設
請參考 `DeepfakeWeb`

# 函式庫版本
**為上述資料處理、模型訓練、使用者介面等會用到之函式庫**  
請參考 `requirements.txt`  
注 : Pytorch版本不同很可能會使模型無法如期進行訓練
