# 無人機飛行載具之智慧計數

## 1. 安裝YOLOv7

```
git clone https://github.com/WongKinYiu/yolov7.git

cd yolov7
pip install -r requirements.txt
```

```
AI-CUP-DRONE
├── datasets/DRONE
│   ├── drone.yaml
│   ├── mutiscale.py
│   ├── resample.py
│   ├── split_train_val.py
│   |   ...
├── yolov7
├── 2ans.py
|   ...
```

## 2. 準備訓練資料
安裝環境
```
pip install opencv-python albumentations
```

下載資料集放在datasets/DRONE/train下
```
datasets/DRONE/train
├── images
│   ├── img0001.png
│   ├── img0002.png
│   |   ...
├── labels
│   ├── img0001.txt
│   ├── img0002.txt
│   |   ...
|   ...
```

```
python resample.py
python mutiscale.py
python split_train_val.py
```

改寫drone.yml
```yaml
# path
train: ../datasets/DRONE/yolov7format_mutiscale/images/train
val: ../datasets/DRONE/yolov7format_mutiscale/images/val

# number of classes
nc: 4

# class names
names: [ car, hov, person, motorcycle]

```

## 3. Train
```
cd ../../yolov7
python train.py --data ../datasets/DRONE/drone.yaml --epochs 100 --batch-size 8 --img-size 1280 --workers 4 --name drone_mutiscale
```

## 4. detect
```
python detect.py --weights runs/train/drone/weights/best.pt --source ../datasets/DRONE/test/ --img-size 1280 --device 3 --name drone --save-txt

cd ../
python 2ans.py
```
