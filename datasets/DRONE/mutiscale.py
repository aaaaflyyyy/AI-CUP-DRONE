import albumentations as A
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import shutil

class Crop:
    def __init__(self,x_min, y_min, x_max, y_max):
        self.transform = A.Compose([
            A.Crop(x_min, y_min, x_max, y_max)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, im, bboxes, class_labels):
        new = self.transform(image=im, bboxes=bboxes, class_labels=class_labels)  # transformed
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels

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

yolopath = './yolov7format_mutiscale'
os.makedirs(yolopath, exist_ok=True)
os.makedirs(yolopath + '/images', exist_ok=True)
os.makedirs(yolopath + '/labels', exist_ok=True)
# os.makedirs(yolopath + '/images/train', exist_ok=True)
# os.makedirs(yolopath + '/labels/train', exist_ok=True)
# os.makedirs(yolopath + '/images/val', exist_ok=True)
# os.makedirs(yolopath + '/labels/val', exist_ok=True)

image_paths = sorted(glob('./yolov7format_resample/images/*/*.png'))

def crop(image, bboxes, class_labels, left, top, right, bottom):
    Crop_img = Crop(left, top, right, bottom)
    crop_image, crop_bboxex= Crop_img(image, bboxes, class_labels) 

    return crop_image, crop_bboxex

def resize(image, bboxes, class_labels, height, width):
    Resize_img = Resize(height, width)
    resize_image, resize_bboxex = Resize_img(image, bboxes, class_labels) 

    return resize_image, resize_bboxex
    
def resameple(filename, image, bboxes, class_labels):
    height, width, _= image.shape
    left = 0
    top = 0
    while left < width:
        if left + 1280 >= width:
            left = max(width - 1280, 0)

        top = 0
        while top < height:
            if top + 1280 >= height:
                top = max(height - 1280, 0)

            right = min(left + 1280, width-1)
            bottom = min(top + 1280, height-1)

            crop_image, crop_bboxex = crop(image, bboxes, class_labels, left, top, right, bottom)

            cv2.imwrite(f'./{yolopath}/images/{filename}_{left}_{top}.png', crop_image)
            with open(f'./{yolopath}/labels/{filename}_{left}_{top}.txt','w') as fw:
                for bbox in crop_bboxex:
                    cls, x1, y1, box_w, box_h = bbox
                    if box_w < 1 or box_h < 1:
                        print(filename)
                        continue

                    x = (x1 + box_w / 2.) / crop_image.shape[1]
                    y = (y1 + box_h / 2.) / crop_image.shape[0]
                    w = box_w / crop_image.shape[1]
                    h = box_h / crop_image.shape[0]
                    
                    fw.write(f'{int(cls)} {x:.06f} {y:.06f} {w:.06f} {h:0.6f}\n')
                fw.close()

            if (top + 1280 >= height):
                    break
            else:
                top = top + 960
        if (left + 1280 >= width):
                break
        else:
            left = left + 960

# {(720, 1344), (1080, 1920)}
for image_path in tqdm(image_paths):
    filename = image_path.split('\\')[-1].split('.png')[0]
    dir = image_path.split('\\')[-2]
    
    image = cv2.imread(image_path)
    height, width, _= image.shape

    bboxes = []
    class_labels = []
    
    with open(f'./yolov7format_v1/labels/{dir}/{filename}.txt','r') as fr:
        for line in fr.readlines():
            cls, x, y, w, h = [float(i) for i in line.strip().split(' ')]
            
            x1 = int((x-w/2) * width)
            y1 = int((y-h/2) * height)
            box_w = int(w * width)
            box_h = int(h * height)

            if box_w < 1 or box_h < 1:
                print(filename)
                continue

            bboxes.append([x1, y1, box_w, box_h])
            class_labels.append(int(cls))
        fr.close()

    for scale in [1, 1.5, 2]:
        resize_image, resize_bboxex = resize(image, bboxes, class_labels, int(height*scale), int(width*scale))

        # cv2.imwrite(f'./yolov7format_resample/images/train/{filename}_{int(scale*100)}.png', resize_image)
        # with open(f'./yolov7format_resample/labels/train/{filename}_{int(scale*100)}.txt','w') as fw:
        #     for bbox in resize_bboxex:
        #         cls, x1, y1, box_w, box_h = bbox

        #         x = (x1 + box_w / 2.) / resize_image.shape[1]
        #         y = (y1 + box_h / 2.) / resize_image.shape[0]
        #         w = box_w / resize_image.shape[1]
        #         h = box_h / resize_image.shape[0]
                
        #         fw.write(f'{int(cls)} {x:.06f} {y:.06f} {w:.06f} {h:0.6f}\n')
        #     fw.close()
        
        resize_bboxex = resize_bboxex.astype(np.int32)
        resize_bboxex_s = []
        class_labels_s = []

        for bbox in resize_bboxex:
            cls, x1, y1, box_w, box_h = bbox
            if box_w < 1 or box_h < 1:
                print(filename)
                continue
            resize_bboxex_s.append([x1, y1, box_w, box_h])
            class_labels_s.append(int(cls))

        resameple(f'{filename}_{int(scale*100)}', resize_image, resize_bboxex_s, class_labels_s)