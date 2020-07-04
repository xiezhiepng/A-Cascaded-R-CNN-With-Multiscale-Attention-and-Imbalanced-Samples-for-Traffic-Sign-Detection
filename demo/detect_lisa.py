#!/usr/bin/python
#coding:utf-8
from mmdet.apis import init_detector, inference_detector, show_result

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
#gtsdb
#config_file = '../configs/libra_rcnn/GTSDB/2_cascade_rcnn_r50_fpn_1x.py'
#checkpoint_file = '../work_dirs/2_GTSDB/epoch_11.pth'

#lisa
config_file = '../configs/libra_rcnn/Lisa/2_cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = '../Lisa/20_test/nanli_attention/epoch_2.pth'

#config_file = '../configs/libra_rcnn/TITS/Ablation/attention_nanli_cascade_rcnn_r50_fpn_1x.py'
#checkpoint_file = '../work_dirs/Ablation/attention_nanli_cascade_rcnn_r50_fpn_1x/epoch_11.pth'
# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
#img = './gtsdb/00004.jpg'
#result = inference_detector(model, img)
#gtsdb
#classes = ( 'danger','mandatory','prohibitory')#,'mandatory', 'danger')
#print(model.CLASSES)
classes = ('warning','speedLimit','noTurn','stop')
#show_result(img, result, classes)#model.CLASSES)

# 测试一系列图片
data_root = './gtsdb/'
#gtsdb
#imgs = ['./gtsdb/00096.jpg']#,'./gtsdb/00002.jpg','./gtsdb/00041.jpg','./gtsdb/00052.jpg','./gtsdb/00095.jpg','./gtsdb/00096.jpg','./gtsdb/00108.jpg','./gtsdb/00112.jpg']
#ccstsdb
imgs = ['./lisa/noLeftTurn_1323803031.avi_image0.jpg','./lisa/noLeftTurn_1323803031.avi_image6.jpg','./lisa/noLeftTurn_1333396099.avi_image0.jpg','./lisa/noRightTurn_1323804170.avi_image2.jpg','./lisa/noUTurn_1333393332.avi_image5.jpg','./lisa/noUTurn_1333393332.avi_image7.jpg','./lisa/noUTurn_1333393332.avi_image6.jpg','./lisa/noUTurn_1333393327.avi_image1.jpg','./lisa/noUTurn_1333393332.avi_image1.jpg','./lisa/noUTurn_1333393391.avi_image6.jpg','./lisa/speedLimit35_1333396407.avi_image10.jpg','./lisa/speedLimit30_1333395404.avi_image1.jpg','./lisa/noUTurn_1333393391.avi_image3.jpg','./lisa/stopAhead_1323824752.avi_image4.jpg','./lisa/yieldAhead_1333395674.avi_image7.jpg','./lisa/turnRight_1323820783.avi_image23.jpg','./lisa/zoneAhead45_1333394398.avi_image0.jpg']#,'./gtsdb/00002.jpg','./gtsdb/00041.jpg','./gtsdb/00052.jpg','./gtsdb/00095.jpg','./gtsdb/00096.jpg','./gtsdb/00108.jpg','./gtsdb/00112.jpg']
for i, result in enumerate(inference_detector(model,imgs)):
    show_result(imgs[i], result,classes)
    #show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
