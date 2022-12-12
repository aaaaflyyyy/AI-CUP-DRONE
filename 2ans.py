from glob import glob
from tqdm import tqdm
from datetime import datetime
import cv2


txt_files = glob('./yolov7/runs/detect/drone_mutiscale/labels/*.txt')

output_file = open(f'result_{datetime.now().month:02d}{datetime.now().day:02d}_drone_mutiscale.csv', 'w')

for txt_file in tqdm(txt_files):
    filename = txt_file.split('\\')[-1].split('.txt')[0]

    img = cv2.imread(f'./datasets/DRONE/test/{filename}.png')
    height, width, _= img.shape

    with open(txt_file, 'r') as fr:
        for line in fr.readlines():
            c, x, y, w, h = [eval(x) for x in line.strip().split(' ')]

            x1 = int((x-w/2) * width)
            y1 = int((y-h/2) * height)
            box_w = int(w * width)
            box_h = int(h * height)

            # img = cv2.rectangle(img, (x1,y1), (x1+box_w,y1+box_h),(0,255,0),3)

            # print(f'{filename},{c},{x1},{y1},{box_w},{box_h}')
            output_file.write(f'{filename},{c},{x1},{y1},{box_w},{box_h}\n')

        # cv2.imshow('i',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()