from glob import glob
import random
from tqdm import tqdm
import shutil
import os

yolopath = './yolov7format_v1_multiscale'
os.makedirs(yolopath, exist_ok=True)
os.makedirs(yolopath + '/images', exist_ok=True)
os.makedirs(yolopath + '/labels', exist_ok=True)
os.makedirs(yolopath + '/images/train', exist_ok=True)
os.makedirs(yolopath + '/labels/train', exist_ok=True)
os.makedirs(yolopath + '/images/val', exist_ok=True)
os.makedirs(yolopath + '/labels/val', exist_ok=True)

all_paths = glob(f'./{yolopath}/images/*.png')
n_file = len(all_paths)

n_train = int(n_file * 0.8)
n_val = n_file - n_train

print(f'train files: {n_train}')
print(f'val file: {n_val}')
random.shuffle(all_paths)

class_cnt = [0, 0, 0, 0]

for image_path in tqdm(all_paths[n_train:]):
    filename = image_path.split('\\')[-1].split('.png')[0]
    txtsrc = f'./{yolopath}/labels/{filename}.txt'
    with open(txtsrc, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            c, xmin, ymin, box_w, box_h = [float(i) for i in line.strip().split(' ')]
            class_cnt[int(c)] += 1
        fr.close()
print(class_cnt)

s = input('split? [y/n]')
if s == 'n':
    assert False

else:
    pass

for image_path in tqdm(all_paths[:n_train]):
    filename = image_path.split('\\')[-1].split('.png')[0]
    
    imgsrc = f'./{yolopath}/images/{filename}.png'
    imgdst = f'./{yolopath}/images/train/{filename}.png'

    shutil.move(imgsrc, imgdst)

    txtsrc = f'./{yolopath}/labels/{filename}.txt'
    txtdst = f'./{yolopath}/labels/train/{filename}.txt'

    shutil.move(txtsrc, txtdst)


for image_path in tqdm(all_paths[n_train:]):
    filename = image_path.split('\\')[-1].split('.png')[0]
    
    imgsrc = f'./{yolopath}/images/{filename}.png'
    imgdst = f'./{yolopath}/images/val/{filename}.png'

    shutil.move(imgsrc, imgdst)

    txtsrc = f'./{yolopath}/labels/{filename}.txt'
    txtdst = f'./{yolopath}/labels/val/{filename}.txt'

    shutil.move(txtsrc, txtdst)
