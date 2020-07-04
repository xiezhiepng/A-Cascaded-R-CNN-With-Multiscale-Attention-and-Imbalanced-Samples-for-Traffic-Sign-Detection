#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:30:59 2019

@author: XZP
把pascal数据集转成coco数据集
"""
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
 
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
 
 
class voc2coco:
    def __init__(self, devkit_path=None, year=None):
#        self.classes = ('__background__',  'pedestrian_crossing','pass_right_side', 'no_stopping_no_standing','50_sign','70_sign','80_sign','100_sign','priority_road','give_way','no_parking')
#        
        self.classes = ('__background__', 'prohibitory', 'mandatory', 'danger')
        
#        self.classes = ('__background__', 'prohibitory', 'mandatory', 'warning')  ,'MISC_SIGNS','PASS_EITHER_SIDE','90_SIGN','OTHER','60_SIGN','30_SIGN','PASS_LEFT_SIDE','110_SIGN','STOP','URDBL','120_SIGN'
        
        
        
#        self.classes = ('__background__',  'PEDESTRIAN_CROSSING','PASS_RIGHT_SIDE', 'NO_STOPPING_NO_STANDING','50_SIGN','70_SIGN','80_SIGN','100_SIGN','PRIORITY_ROAD','GIVE_WAY','NO_PARKING','MISC_SIGNS','URDBL')
        
        #,'MISC_SIGNS','PASS_EITHER_SIDE','90_SIGN','OTHER','60_SIGN','30_SIGN','PASS_LEFT_SIDE','110_SIGN','STOP','URDBL','120_SIGN'
        self.num_classes = len(self.classes)
        assert 'VOCdevkit' in devkit_path, 'VOC地址不存在: {}'.format(devkit_path)
        self.data_path = os.path.join(devkit_path, 'VOC' + year)
        self.annotaions_path = os.path.join(self.data_path, 'Annotations')
        self.image_set_path = os.path.join(self.data_path, 'ImageSets')
        self.year = year
        self.categories_to_ids_map = self._get_categories_to_ids_map()
        self.categories_msg = self._categories_msg_generator()
 
    def _load_annotation(self, ids=[]):
        ids = ids if _isArrayLike(ids) else [ids]
        image_msg = []
        annotation_msg = []
        annotation_id = 1
        for index in ids:
            filename = '{:0>5}'.format(index)
            json_file = os.path.join(self.data_path, 'Segmentation_json', filename + '.json')
            if os.path.exists(json_file):
                img_file = os.path.join(self.data_path, 'JPEGImages', filename + '.jpg')
                im = cv2.imread(img_file)
                width = im.shape[1]
                height = im.shape[0]
                seg_data = json.load(open(json_file, 'r'))
                assert type(seg_data) == type(dict()), 'annotation file format {} not supported'.format(type(seg_data))
                for shape in seg_data['shapes']:
                    seg_msg = []
                    for point in shape['points']:
                        seg_msg += point
                    one_ann_msg = {"segmentation": [seg_msg],
                                   "area": self._area_computer(shape['points']),
                                   "iscrowd": 0,
                                   "image_id": int(index),
                                   "bbox": self._points_to_mbr(shape['points']),
                                   "category_id": self.categories_to_ids_map[shape['label']],
                                   "id": annotation_id,
                                   "ignore": 0
                                   }
                    annotation_msg.append(one_ann_msg)
                    annotation_id += 1
            else:
                xml_file = os.path.join(self.annotaions_path, filename + '.xml')
                print("self.annotaions_path, filename + '.xml':"+filename+'.xml')
                tree = ET.parse(xml_file)
                size = tree.find('size')
                objs = tree.findall('object')
                width = size.find('width').text
                height = size.find('height').text
                for obj in objs:
                    bndbox = obj.find('bndbox')
                    [xmin, xmax, ymin, ymax] \
                        = [round(float(bndbox.find('xmin').text)) - 1, round(float(bndbox.find('xmax').text)),
                           round(float(bndbox.find('ymin').text)) - 1, round(float(bndbox.find('ymax').text))]
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    bbox = [xmin, xmax, ymin, ymax]
                    one_ann_msg = {"segmentation": self._bbox_to_mask(bbox),
                                   "area": self._bbox_area_computer(bbox),
                                   "iscrowd": 0,
                                   "image_id": str(index),
                                   "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                                   "category_id": self.categories_to_ids_map[obj.find('name').text],
                                   "id": annotation_id,
                                   "ignore": 0
                                   }
                    annotation_msg.append(one_ann_msg)
                    annotation_id += 1
            one_image_msg = {"file_name": filename + ".jpg",
                             "height": int(height),
                             "width": int(width),
                             "id": str(index)
                             }
            image_msg.append(one_image_msg)
        return image_msg, annotation_msg
    def _bbox_to_mask(self, bbox):
        assert len(bbox) == 4, 'Wrong bndbox!'
        mask = [bbox[0], bbox[2], bbox[0], bbox[3], bbox[1], bbox[3], bbox[1], bbox[2]]
        return [mask]
    def _bbox_area_computer(self, bbox):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        return width * height
    def _save_json_file(self, filename=None, data=None):
        json_path = os.path.join(self.data_path, 'cocoformatJson')
        assert filename is not None, 'lack filename'
        if os.path.exists(json_path) == False:
            os.mkdir(json_path)
        if not filename.endswith('.json'):
            filename += '.json'
        assert type(data) == type(dict()), 'data format {} not supported'.format(type(data))
        with open(os.path.join(json_path, filename), 'w') as f:
            f.write(json.dumps(data))
    def _get_categories_to_ids_map(self):
        return dict(zip(self.classes, range(self.num_classes)))
    def _get_all_indexs(self):
        ids = []
        for root, dirs, files in os.walk(self.annotaions_path, topdown=False):
            for f in files:
                if str(f).endswith('.xml'):
                    id = int(str(f).strip('.xml'))
                    ids.append(id)
        assert ids is not None, 'There is none xml file in {}'.format(self.annotaions_path)
        return ids
    def _get_indexs_by_image_set(self, image_set=None):
        if image_set is None:
            return self._get_all_indexs()
        else:
            image_set_path = os.path.join(self.image_set_path, 'Main', image_set + '.txt')
            assert os.path.exists(image_set_path), 'Path does not exist: {}'.format(image_set_path)
            with open(image_set_path) as f:
                ids = [x.strip() for x in f.readlines()]
            return ids
    def _points_to_mbr(self, points):
        assert _isArrayLike(points), 'Points should be array like!'
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        assert len(x) == len(y), 'Wrong point quantity'
        xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
        height = ymax - ymin
        width = xmax - xmin
        return [xmin, ymin, width, height]
    def _categories_msg_generator(self):
        categories_msg = []
        for category in self.classes:
            if category == '__background__':
                continue
            one_categories_msg = {"supercategory": "none",
                                  "id": self.categories_to_ids_map[category],
                                  "name": category
                                  }
            categories_msg.append(one_categories_msg)
        return categories_msg
    def _area_computer(self, points):
        assert _isArrayLike(points), 'Points should be array like!'
        tmp_contour = []
        for point in points:
            tmp_contour.append([point])
        contour = np.array(tmp_contour, dtype=np.int32)
        area = cv2.contourArea(contour)
        return area
    def voc_to_coco_converter(self):
        img_sets = ['trainval']# 'val',
        for img_set in img_sets:
            
            ids = self._get_indexs_by_image_set(img_set)
            print("img_set:"+img_set)
            img_msg, ann_msg = self._load_annotation(ids)
            
            result_json = {"images": img_msg,
                           "type": "instances",
                           "annotations": ann_msg,
                           "categories": self.categories_msg}
            self._save_json_file('voc_' + self.year + '_' + img_set, result_json)
def demo():
    # 转换pascal地址是'./VOC2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    converter = voc2coco('VOCdevkit/', '2007')
#    converter = voc2coco('/Volumes/文档/untitledfolder/VOCdevkit/', '2007')
#    converter = voc2coco('/Volumes/文档/untitledfolder/CCTSDB/training_data/VOCdevkit/', '2007')
#    converter = voc2coco('/Volumes/文档/学习资料/交通标志检测数据集/中科院交通标志数据集/CTSB/VOCdevkit/', '2007')
    converter.voc_to_coco_converter()
if __name__ == "__main__":
    demo()
    
    

