from mmdet.apis import init_detector, inference_detector, show_result

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
#gtsdb
#config_file = '../configs/libra_rcnn/GTSDB/2_cascade_rcnn_r50_fpn_1x.py'
#checkpoint_file = '../work_dirs/2_GTSDB/epoch_11.pth'

#cctsdb
#config_file = '../configs/libra_rcnn/GTSDB/2_cascade_rcnn_r50_fpn_1x.py'
#checkpoint_file = '../work_dirs/2_GTSDB/epoch_2.pth'

config_file = '../configs/libra_rcnn/GTSDB/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = '../work_dirs/demo_cascadercnn/epoch_12.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
#img = './gtsdb/00004.jpg'
#result = inference_detector(model, img)
#gtsdb
#classes = ( 'danger','mandatory','prohibitory')#,'mandatory', 'danger')
#print(model.CLASSES)
classes = ('prohibitory','mandatory','warning')# 'danger')
#show_result(img, result, classes)#model.CLASSES)

# 测试一系列图片
#data_root = './gtsdb/'
#gtsdb
#imgs = ['./gtsdb/00096.jpg']#,'./gtsdb/00002.jpg','./gtsdb/00041.jpg','./gtsdb/00052.jpg','./gtsdb/00095.jpg','./gtsdb/00096.jpg','./gtsdb/00108.jpg','./gtsdb/00112.jpg']
#ccstsdb

def add_imgpath(path,imgs):
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirpath,filename))
            imgs.append(os.path.join(dirpath,filename))
import os
path =r"cctsdb/"
imgs = []
add_imgpath(path,imgs)
#imgs = ['./cctsdb/nan1.jpg','./gtsdb/00004.jpg','./gtsdb/00041.jpg','./gtsdb/00052.jpg','./gtsdb/00095.jpg','./gtsdb/00096.jpg','./gtsdb/00108.jpg','./gtsdb/00112.jpg']
for i, result in enumerate(inference_detector(model,imgs)):
    show_result(imgs[i], result,classes)
    #show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
