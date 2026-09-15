# Vehicle Recognition System

新的遙控甩尾分析專案，將流程拆成可重複使用的階段：

1. 匯入影片
2. 以可設定幀間隔抽取車輛軌跡
3. 輸出 `trajectory.png` 與 `trajectory.json`
4. 用保存的軌跡資料重算評分，不必重新跑模型
5. 跑道、車輛標註與模型訓練各自獨立

## 安裝

```powershell
C:/Users/Local.User/AppData/Local/Programs/Python/Python311/python.exe -m pip install -e .
```

## 產生軌跡

```powershell
vehicle-rec extract --video C:/path/to/video.mp4 --output runs/demo --sample-frames 5
```

`runs/demo` 會產生：

- `trajectory.png`: 所有追蹤路線的視覺化圖片
- `trajectory.json`: 每台車的 frame、time、中心點、外框與角度
- `metadata.json`: 影片 FPS、尺寸與抽樣設定

## 重算評分

```powershell
vehicle-rec score --trajectory runs/demo/trajectory.json --output runs/demo/score.json
```

評分目前固定為：路線 40、角度 20、速度 10、穩定度 30（速度穩定 10、角度穩定 20）。

## 模組邊界

- `trajectory.py`: 影片與模型推論、軌跡輸出
- `scoring.py`: 純資料評分，可獨立測試與調參
- `models.py`: JSON 資料結構
- 後續可新增 `zones.py` 與 `training.py`，不影響評分器
