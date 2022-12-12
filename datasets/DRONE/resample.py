import albumentations as A
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import shutil

class HorizontalFlip:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=1)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, im, bboxes, class_labels):
        new = self.transform(image=im, bboxes=bboxes, class_labels=class_labels)  # transformed
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels

def hf(image, bboxes, class_labels):
    hf_img = HorizontalFlip()
    hf_image, hf_bboxex= hf_img(image, bboxes, class_labels) 

    return hf_image, hf_bboxex

class Resize:
    def __init__(self,height, width):
        self.transform = A.Compose([
            A.Resize(height, width, cv2.INTER_CUBIC)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, im, bboxes, class_labels):
        new = self.transform(image=im, bboxes=bboxes, class_labels=class_labels)  # transformed
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels

yolopath = './yolov7format_resample'
os.makedirs(yolopath, exist_ok=True)
os.makedirs(yolopath + '/images', exist_ok=True)
os.makedirs(yolopath + '/labels', exist_ok=True)
# os.makedirs(yolopath + '/images/train', exist_ok=True)
# os.makedirs(yolopath + '/labels/train', exist_ok=True)
# os.makedirs(yolopath + '/images/val', exist_ok=True)
# os.makedirs(yolopath + '/labels/val', exist_ok=True)

with open('person_more_img.txt', 'r') as fr:
    for txt_path in fr.readlines():
        filename = txt_path.strip().split('\\')[-1].split('.txt')[0]

        image = cv2.imread(f'./train/images/{filename}.png')
        shutil.copy(f'./train/images/{filename}.png', f'./{yolopath}/images/{filename}.png')
        height, width, _= image.shape

        with open(f'./train/labels/{filename}.txt','r') as fr:
            with open(f'./{yolopath}/labels/{filename}.txt','w') as fw:
                for line in fr.readlines():
                    cls, x1, y1, box_w, box_h = [int(i) for i in line.strip().split(',')]
                    if box_w < 1 or box_h < 1:
                        print(filename)
                        continue

                    x = (x1 + box_w / 2.) / image.shape[1]
                    y = (y1 + box_h / 2.) / image.shape[0]
                    w = box_w / image.shape[1]
                    h = box_h / image.shape[0]
                    
                    fw.write(f'{int(cls)} {x:.06f} {y:.06f} {w:.06f} {h:0.6f}\n')
                fw.close()
                fr.close()
        # bboxes = []
        # class_labels = []
        
        # with open(f'./train/labels/{filename}.txt','r') as fr:
        #     for line in fr.readlines():
        #         cls, x1, y1, box_w, box_h = [int(i) for i in line.strip().split(',')]
        #         if box_w < 1 or box_h < 1:
        #             print(filename)
        #             continue
        #         bboxes.append([x1, y1, box_w, box_h])
        #         class_labels.append(cls)
        #     fr.close()

        # hf_image, hf_bboxex = hf(image, bboxes, class_labels)
        # cv2.imwrite(f'./{yolopath}/images/{filename}_flip.png', hf_image)
        # with open(f'./{yolopath}/labels/{filename}_flip.txt','w') as fw:
        #     for bbox in hf_bboxex:
        #         cls, x1, y1, box_w, box_h = bbox
        #         if box_w < 1 or box_h < 1:
        #             print(filename)
        #             continue

        #         x = (x1 + box_w / 2.) / hf_image.shape[1]
        #         y = (y1 + box_h / 2.) / hf_image.shape[0]
        #         w = box_w / hf_image.shape[1]
        #         h = box_h / hf_image.shape[0]
                
        #         fw.write(f'{int(cls)} {x:.06f} {y:.06f} {w:.06f} {h:0.6f}\n')
        #     fw.close()

with open('car_more_img.txt', 'r') as fr:
    for txt_path in fr.readlines():
        filename = txt_path.strip().split('\\')[-1].split('.txt')[0]

        image = cv2.imread(f'./train/images/{filename}.png')
        shutil.copy(f'./train/images/{filename}.png', f'./{yolopath}/images/{filename}.png')
        height, width, _= image.shape

        with open(f'./train/labels/{filename}.txt','r') as fr:
            with open(f'./{yolopath}/labels/{filename}.txt','w') as fw:
                for line in fr.readlines():
                    cls, x1, y1, box_w, box_h = [int(i) for i in line.strip().split(',')]
                    if box_w < 1 or box_h < 1:
                        print(filename)
                        continue

                    x = (x1 + box_w / 2.) / image.shape[1]
                    y = (y1 + box_h / 2.) / image.shape[0]
                    w = box_w / image.shape[1]
                    h = box_h / image.shape[0]
                    
                    fw.write(f'{int(cls)} {x:.06f} {y:.06f} {w:.06f} {h:0.6f}\n')
                fw.close()
                fr.close()
