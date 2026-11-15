# Lab 3：胸腔 X 光四分類（NYCU AIMI 2026）

把兒童／成人胸腔 X 光分成 normal、bacteria、virus、COVID-19 四類，評分指標為 macro-F1。

最終提交是三組 ImageNet 預訓練 ResNet18 的集成（R5＋R8＋R9，共 15 個模型的機率平均）：5-fold OOF macro-F1 0.887，Kaggle public 0.877。只使用官方提供的資料，沒有使用外部 CXR 資料或 CXR 預訓練權重。

## 環境

Python 3.12、torch 2.11.0+cu128、torchvision 0.26.0+cu128（版本鎖定在 `requirements.txt`）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

## 資料

競賽資料不包含在 repo 中。從 Kaggle 下載後解壓到 `data/raw/`：

```
data/raw/train_images/   data/raw/val_images/   data/raw/test_images/
data/raw/train_data.csv  data/raw/val_data.csv
test_data_sample.csv     （放在專案根目錄）
```

```powershell
.\.venv\Scripts\python.exe check_data.py          # 檢查筆數、標籤、解碼
.\.venv\Scripts\python.exe prepare_data.py        # EDA、重複影像分組、5-fold 切分 -> data/folds.csv
.\.venv\Scripts\python.exe prepare_input.py 512   # 長邊 512 的影像快取 -> data/cache/long512_npy
.\.venv\Scripts\python.exe prepare_input.py 768   # 長邊 768 的影像快取
```

快取建議放在 SSD；若放在其他位置，用環境變數 `LAB3_CACHE_ROOT` 指定（目錄下為 `long512_npy`、`long768_npy`）。

## 重現最終提交

```powershell
.\.venv\Scripts\python.exe train.py --run R5_geocolor_p12 --model resnet18 --size 512 --bs 24 --aug geo_color --patience 12 --epochs 40
.\.venv\Scripts\python.exe train.py --run R8_sam --model resnet18 --size 512 --bs 24 --aug geo_color --patience 12 --epochs 40 --sam 0.05
.\.venv\Scripts\python.exe train.py --run R9_sam_768 --model resnet18 --size 768 --bs 24 --patience 12 --epochs 40 --sam 0.05
.\.venv\Scripts\python.exe predict.py --runs R5_geocolor_p12 R8_sam R9_sam_768 --name ENS3_R5_R8_R9
```

RTX 4070 上三組分別約 62、76、129 分鐘。`predict.py` 會檢查提交檔格式（欄位、1,182 列、順序與 sample 一致、每列恰好一個 1），且不覆寫已存在的檔案。

## 檔案

| 檔案 | 用途 |
|---|---|
| `train.py` | 5-fold 訓練：letterbox、增強、SAM、early stopping；輸出 `outputs/<run>/`（OOF 機率、metrics、fold 權重），並在 `experiments.csv` 新增一列 |
| `predict.py` | 平均多個 run 的 fold 模型機率，寫出 `submissions/<name>.csv` |
| `prepare_data.py` | EDA、dHash 重複影像偵測、StratifiedGroupKFold 切分 |
| `prepare_input.py` | 影像快取（長邊縮放、不放大、記錄 SHA-256） |
| `analyze_errors.py`、`gradcam.py`、`tta.py`、`hierarchy.py` | 錯誤分析、Grad-CAM、TTA 與分層分類實驗 |
| `report_figures.py` | 報告用的彙整圖 |
| `run_overnight.ps1`、`run_backbones.ps1` | 依序執行多個實驗的佇列（已完成的會跳過） |
| `experiments.csv` | 全部實驗的 OOF 分數與 Kaggle public 分數 |

主要指標是全部 4,726 張 train＋val 影像的 OOF macro-F1。public 排行榜只有約 591 張，雜訊太大，不用來挑模型。
