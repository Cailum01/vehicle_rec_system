# Vehicle Recognition System 使用教學手冊

## 1. 系統用途

本系統用於分析遙控甩尾影片，將影片中的車輛偵測結果整理成可重複使用的軌跡資料，並依保存的軌跡計算分數。

啟動程式後可以直接從選單選擇模式，也可以使用進階 CLI 指令。選單提供三個主要方向：評分、訓練跑道、訓練車輛。

目前提供六個 CLI 功能：

1. `extract`：讀取影片，使用 Ultralytics YOLO 追蹤車輛，輸出軌跡 JSON、視覺化 PNG 與影片 metadata。
2. `score`：讀取既有的軌跡 JSON 計算分數，不重新讀取影片，也不重新執行模型。
3. `track-label`：標註護欄、跑道與得分區，建立 segmentation 訓練資料。
4. `track-train`：訓練跑道 segmentation 模型。
5. `vehicle-label`：標註遙控車輪廓，建立 vehicle segmentation 訓練資料。
6. `vehicle-train`：訓練車輛 segmentation 模型。

角度估計目前仍未由影片抽取流程產生；跑道標註、車輛標註、模型訓練與評分流程已可由 CLI 或互動選單操作。

## 2. 安裝與環境

### 2.1 必要條件

- Windows、PowerShell
- Python 3.11 或更新版本
- 可讀取的影片檔案，例如 `.mp4`
- Ultralytics YOLO 模型檔案，例如自訂的 `.pt` 檔
- 模型所需的執行環境，例如 CPU 或相容的 GPU/CUDA 設定

專案依賴會由 `pyproject.toml` 安裝：

- `numpy>=1.26`
- `opencv-python>=4.9`
- `ultralytics>=8.3`

### 2.2 安裝專案

在專案根目錄執行：

```powershell
python -m pip install -e .
```

README 中也提供了指定 Python 3.11 執行檔的寫法：

```powershell
C:/Users/Local.User/AppData/Local/Programs/Python/Python311/python.exe -m pip install -e .
```

安裝後可確認 CLI 已註冊：

```powershell
vehicle-rec --help
```

若 CLI 尚未加入 PATH，可使用模組方式執行：

```powershell
python -m vehicle_rec_system.cli --help
```

### 2.3 最簡單的操作方式

不帶任何參數啟動程式：

```powershell
vehicle-rec
```

或使用：

```powershell
python -m vehicle_rec_system.cli
```

畫面會顯示：

```text
=== Vehicle Recognition System ===
1. 評分
2. 訓練
0. 離開
```

選擇 `2. 訓練` 後，再選擇：

```text
1. 訓練跑道
2. 訓練車輛
```

程式會逐步詢問影片路徑、資料夾、模型與訓練回合數；直接按 Enter 可使用顯示的預設值。

## 3. CLI 總覽

```text
vehicle-rec <command> [options]

commands:
  extract   從影片抽取車輛軌跡
  score     使用既有軌跡計算分數
  track-label   標註護欄、跑道與得分區
  track-train   訓練跑道 segmentation 模型
  vehicle-label 標註車輛輪廓
  vehicle-train 訓練車輛 segmentation 模型
```

兩個命令都支援 `--help`：

```powershell
vehicle-rec extract --help
vehicle-rec score --help
vehicle-rec track-label --help
vehicle-rec track-train --help
vehicle-rec vehicle-label --help
vehicle-rec vehicle-train --help
```

## 4. 從零開始訓練跑道與車輛

以下流程全部在目前的 `vehicle_rec_system` 專案中執行，不需要使用舊版 `Vehicle_Recognition` 程式。

### 4.1 標註跑道

```powershell
vehicle-rec track-label `
  --video C:/data/track.mp4 `
  --dataset data/track_dataset `
  --sample-frames 10
```

標註視窗操作：

- `1`：切換 `wall` 護欄類別
- `2`：切換 `track` 跑道類別
- `3`：切換 `score_zone` 得分區類別
- 滑鼠左鍵：新增多邊形節點
- `Enter`：完成目前多邊形
- `N`：保存目前畫面
- `U`：撤銷上一點或上一個多邊形
- `C`：清除目前畫面標註
- `S`：跳過目前畫面
- `Q`：結束標註

按 `N` 保存至少 20 至 50 個涵蓋不同彎道、角度與光線的畫面。資料會建立：

```text
data/track_dataset/
├── data.yaml
├── images/train/
└── labels/train/
```

### 4.2 訓練跑道模型

```powershell
vehicle-rec track-train `
  --dataset data/track_dataset `
  --output runs/track_training `
  --base-model yolo11n-seg.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 4 `
  --name track
```

完成後模型通常位於：

```text
runs/track_training/track/weights/best.pt
```

若顯示記憶體不足，將 `--batch` 改為 `1`；若訓練時間太長，可先用 `--epochs 20` 做功能測試。

### 4.3 標註車輛

```powershell
vehicle-rec vehicle-label `
  --video C:/data/drift.mp4 `
  --dataset data/vehicle_dataset `
  --sample-frames 10
```

車輛標註使用同一套多邊形操作，但只有一個類別 `vehicle`：滑鼠逐點描繪每台車，按 `Enter` 完成多邊形，按 `N` 保存畫面。建議至少保存 30 至 100 個包含不同距離、方向、遮擋與光線的畫面。

### 4.4 訓練車輛模型

```powershell
vehicle-rec vehicle-train `
  --dataset data/vehicle_dataset `
  --output runs/vehicle_training `
  --base-model yolo11n-seg.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 4 `
  --name vehicle
```

訓練後模型通常位於：

```text
runs/vehicle_training/vehicle/weights/best.pt
```

`--base-model` 需要可下載或已存在的 Ultralytics segmentation 模型。若使用的是 detection 模型，必須改用對應的 detection 標註與訓練流程；目前本專案標註器產生的是 segmentation 格式。

## 5. 抽取影片軌跡

### 5.1 指令格式

```powershell
vehicle-rec extract `
  --video <影片路徑> `
  --output <輸出資料夾> `
  --model <YOLO模型路徑> `
  [--sample-frames <正整數>] `
  [--vehicle-class <類別編號>]
```

PowerShell 的反引號 `` ` `` 用來換行；也可以把整個命令寫成單行。

完整範例：

```powershell
vehicle-rec extract `
  --video C:/data/drift.mp4 `
  --output runs/drift-01 `
  --model models/vehicle.pt `
  --sample-frames 5 `
  --vehicle-class 0
```

### 5.2 參數說明

| 參數 | 必要 | 預設值 | 說明 |
|---|---:|---:|---|
| `--video` | 是 | 無 | 輸入影片路徑。OpenCV 必須能開啟此檔案。 |
| `--output` | 是 | 無 | 輸出資料夾；不存在時會自動建立。 |
| `--model` | 是 | 無 | Ultralytics YOLO 模型路徑。這是必要參數，不能省略。 |
| `--sample-frames` | 否 | `5` | 每隔幾個 frame 執行一次追蹤。程式會將小於 1 的值修正為 1。 |
| `--vehicle-class` | 否 | `0` | 要追蹤的模型類別編號。傳入 `0` 時只追蹤 class 0。 |

若要讓模型回傳所有類別，程式內部需要使用 `vehicle_class=None`；目前 CLI 的 `argparse` 介面沒有提供可直接輸入 `None` 的選項，因此 CLI 操作預設是篩選單一類別。

### 5.3 抽取流程

1. 建立輸出資料夾。
2. 使用 OpenCV 開啟影片。
3. 讀取 FPS、寬度與高度。
4. 載入 YOLO 模型。
5. 逐 frame 讀取影片。
6. 只在抽樣 frame 上呼叫 `model.track(..., persist=True)`。
7. 將每個有 track ID 的 bounding box 轉為中心點。
8. 依 track ID 將樣本分組。
9. 在黑色畫布上繪製每台車的中心點與移動線段。
10. 輸出三個檔案。

程式會輸出：

```text
runs/drift-01/
├── trajectory.json
├── metadata.json
└── trajectory.png
```

### 5.4 輸出檔案

#### `trajectory.json`

包含影片資訊與依 track ID 分組的偵測資料：

```json
{
  "video_fps": 30.0,
  "frame_width": 1920,
  "frame_height": 1080,
  "sample_frames": 5,
  "tracks": {
    "1": [
      {
        "frame": 5,
        "time_seconds": 0.1666666667,
        "track_id": 1,
        "bbox": [100, 200, 180, 260],
        "center": [140, 230]
      }
    ]
  }
}
```

欄位意義：

| 欄位 | 說明 |
|---|---|
| `video_fps` | 影片 FPS；若 OpenCV 讀不到，使用 `30.0`。 |
| `frame_width` / `frame_height` | 影片尺寸。 |
| `sample_frames` | 實際用於抽樣的間隔，至少為 1。 |
| `tracks` | 以字串形式保存的 track ID 到樣本陣列的對應。 |
| `frame` | frame 編號。來源程式從 1 開始計數。 |
| `time_seconds` | `frame / FPS`。 |
| `track_id` | YOLO tracker 回傳的追蹤 ID。 |
| `bbox` | `[x1, y1, x2, y2]` 外框座標。 |
| `center` | 外框中心點 `[x, y]`。 |

目前抽取程式不會在樣本中填入 `angle_degrees`；因此這個欄位若要使用，必須由其他工具或人工後處理加入。

#### `metadata.json`

包含原始影片路徑、FPS、尺寸與抽樣設定：

```json
{
  "video": "C:/data/drift.mp4",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "sample_frames": 5
}
```

#### `trajectory.png`

黑色背景的軌跡圖：

- 綠色圓點：偵測到的中心點
- 黃色線段：同一 track 的相鄰中心點之間的移動路線

圖片不是影片疊圖，只呈現抽樣後的中心點軌跡。

## 6. 計算評分

### 6.1 指令格式

```powershell
vehicle-rec score `
  --trajectory <trajectory.json路徑> `
  --output <score.json路徑>
```

範例：

```powershell
vehicle-rec score `
  --trajectory runs/drift-01/trajectory.json `
  --output runs/drift-01/score.json
```

此命令不需要影片與模型，只讀取 `trajectory.json`。

### 6.2 評分輸出

`score.json` 格式如下：

```json
{
  "weights": {
    "route": 40.0,
    "angle": 20.0,
    "speed": 10.0,
    "speed_stability": 10.0,
    "angle_stability": 20.0
  },
  "tracks": {
    "1": {
      "route": 40.0,
      "angle": 0.0,
      "speed": 10.0,
      "speed_stability": 10.0,
      "angle_stability": 20.0,
      "total": 80.0
    }
  }
}
```

目前權重為：

| 項目 | 滿分 | 目前意義 |
|---|---:|---|
| `route` | 40 | 目前只要有至少兩個樣本就給滿分。尚未分析跑道或路線品質。 |
| `angle` | 20 | 依車輛角度與移動方向的平均差異計算。沒有 `angle_degrees` 時為 0。 |
| `speed` | 10 | 依平均像素速度計算，100 像素/秒可達滿分。 |
| `speed_stability` | 10 | 依速度的母體標準差扣分。 |
| `angle_stability` | 20 | 依角度差異的母體標準差扣分。沒有角度資料時為 20。 |

總分為各項相加後限制在 `0` 到 `100`。

### 6.3 重要計算規則

- 速度使用中心點位移距離，不是實際物理速度。
- 速度公式為 `sqrt(dx² + dy²) * FPS / frame_delta`，單位近似為像素/秒。
- 移動方向為 `atan2(dy, dx)` 轉換成 0 到 360 度。
- 角度差會使用最短圓周差，因此 359 度與 1 度的差為 2 度。
- frame 差小於或等於 0 的相鄰樣本會被略過。
- 少於兩個樣本的 track，所有分數都是 `0.0`。
- 輸出數值四捨五入到小數點後兩位。

## 7. 一次完整操作

```powershell
python -m pip install -e .
vehicle-rec extract --video C:/data/drift.mp4 --output runs/demo --model runs/vehicle_training/vehicle/weights/best.pt --track-model runs/track_training/track/weights/best.pt --sample-frames 5 --vehicle-class 0
vehicle-rec score --trajectory runs/demo/trajectory.json --output runs/demo/score.json
Get-Content runs/demo/score.json
```

`--track-model` 會讓抽取器在每個抽樣 frame 判斷車輛是否在跑道、是否進入得分區，以及是否接近護欄。建議先檢查 `trajectory.png` 與 `trajectory.json`，再執行評分。若沒有任何 track，評分輸出仍會成功，但 `tracks` 會是空物件。

## 8. 故障排除

### `Cannot open video`

確認影片路徑正確、檔案存在，且 OpenCV 支援該影片編碼：

```powershell
Test-Path C:/data/drift.mp4
```

### 找不到 `vehicle-rec`

改用模組方式執行，或重新安裝專案：

```powershell
python -m vehicle_rec_system.cli --help
python -m pip install -e .
```

### 找不到模型或模型載入失敗

確認 `--model` 指向有效的 Ultralytics 模型檔案，並確認依賴已安裝：

```powershell
python -m pip install -e .
```

### `trajectory.json` 沒有 track

可能原因包括：模型類別編號不正確、影片中的車輛無法被模型偵測、抽樣間隔過大，或 tracker 沒有產生 ID。可先降低抽樣間隔：

```powershell
vehicle-rec extract --video C:/data/drift.mp4 --output runs/debug --model models/vehicle.pt --sample-frames 1 --vehicle-class 0
```

### 角度分數是 0

目前 `extract` 只輸出 bounding box 與中心點，沒有估計或保存 `angle_degrees`。這是現行實作限制，不是 `score` 命令讀檔失敗。

## 9. 目前限制與使用注意事項

- `--model` 是 CLI 的必要參數；README 的舊範例若省略它，無法直接執行。
- `route` 分數目前不是實際的跑道幾何評分，只代表 track 至少有兩筆樣本。
- `angle_degrees` 的資料結構已被評分器支援，但抽取器尚未產生它。
- `trajectory.png` 的座標原點在左上角，符合 OpenCV 影像座標系。
- `metadata.json` 的 `sample_frames` 直接寫入 CLI 傳入值；若傳入 0 或負數，實際抽樣會修正為 1，但 metadata 可能仍記錄原始值。
- 評分器假設輸入 JSON 至少有 `video_fps`；格式不完整時會產生例外。
