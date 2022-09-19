import os,cv2,keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

class RCNN:
    def __init__(self, dataset:str):
        info, self.dataset= tfds.load(name=dataset, split=('train', 'test'), with_info=True)
        print(f'downloaded {dataset} \n {info}')
        self.show_SS_test()
        # train = self.dataset['train']
        # # unsupervised = self.dataset['unsupervised']
        # print(train)

        # train = train.as_numpy_iterator()
        # data = train.next()

        # img = data['image']
        # cv2.imshow('image',img)
        # cv2.waitKey(0)

        # print(type(data['image']))

        # print(data)

        # rects = self.selective_search(img, use_cv2=True)
        # print(type(rects))
        # print(rects)


    def get_iou(self, bb1: dict, bb2: dict) -> float:
        '''
        Return IoU between bb1 and bb2

        Params
        ----------------
        bb1: dict like {'x1': int, 'x2': int, 'y1': int, 'y2': int}
        bb2: dict like {'x1': int, 'x2': int, 'y1': int, 'y2': int}

        A rectangle is specified with (x1, y1) as the upper left vertex and (x2, y2) as the upper right vertex.

        Description
        -----------------
        IoU is calculated by 'Area of Overlap' divided by 'Area of Union'
        '''

        # rectangle must be specified by (x1, y1): upper_left and (x2, y2): lower_right
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['x2'] < bb2['x2']

        # calculate Overlap coordinates
        overlap_left = max(bb1['x1'], bb2['x1'])
        overlap_right = min(bb1['x2'], bb2['x2'])
        overlap_upper = max(bb1['y1'], bb2['y1'])
        overlap_lower = min(bb1['y2'], bb2['y2'])

        # condition that IoU equals to zero
        area_of_overlap = (overlap_right - overlap_left) * (overlap_lower - overlap_upper)
        if area_of_overlap <= 0:
            return 0.0

        # calculate area of bb1 and bb2, then calculate area of union
        area_of_bb1 = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        area_of_bb2 = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
        area_of_union = area_of_bb1 + area_of_bb2 - area_of_overlap

        IoU = area_of_overlap / area_of_union

        assert 0.0 <= IoU <= 1.0

        return IoU

    def selective_search(self, img: np.ndarray, use_cv2: bool=True) -> np.ndarray:
        '''
        Return RoIs using Selective Search

        Params
        ----------------
        img: np.ndarray of the target image
        use_cv2: bool, if True, use cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() to get RoIs
                       if False, use self-implemented selective search to get RoIs

        Description
        -----------------
        Selective Search is a method to get RoIs (Region of Interest) from an image.
        This time, I use Fast mode of Selective Search.
        return value is a np.ndarray of RoIs, for each RoI, it is specified by [x, y, w, h]
        '''

        if use_cv2:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            return ssresults
        else:
            print('Not implemented yet')
            exit(1)

    def show_SS_test(self):
        print(self.dataset)
        data = self.dataset['train']
        data = data.as_numpy_iterator()
        data = data.next()

        img = data['image']
        label = data['label']
        name = data['filename']

        rects = self.selective_search(img)

        for rect in rects:
            x,y,w,h = rect
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

        cv2.imshow('image',img)
        cv2.waitKey(1)




if __name__ == '__main__':
    dataset = 'kitti'
    rcnn = RCNN(dataset)
