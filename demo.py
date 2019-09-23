# ---------------------------------------------------------
# CNNClassificationTensorRT
# Copyright (c) 2019
# Licensed under The MIT License [see LICENSE for details]
# Written by Rudy Nurhadi
# ---------------------------------------------------------

import os
import sys
import cv2
import time
import numpy as np

from CNNClassificationTensorRT import CNNClassificationTensorRT

CWD_PATH = os.path.dirname(os.path.realpath(__file__))

cnnClassificationTensorRT = CNNClassificationTensorRT(os.path.join(CWD_PATH, 'model'), 'gender_recog', 12, rebuild_engine=False)

def main():
    imgs = []
    for filename in os.listdir('female'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            imgs.append(cv2.imread(os.path.join('female', filename)))
    now = time.time()
    predict_results = cnnClassificationTensorRT.return_predict(imgs)
    for i, result in enumerate(predict_results):
        if result["confidences"][0] > result["confidences"][1]:
            print(result["labels"][0])
        else:
            print(result["labels"][1])
    print("Inference Time: %f" % (time.time() - now))
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
