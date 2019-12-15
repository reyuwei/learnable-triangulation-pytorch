#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import numpy as np
from collections import defaultdict
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower

def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)

def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = R.copy()
        self.t = t.copy()
        self.K = K.copy()
        self.dist = dist

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_width, new_height = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i    

def write_cam(K, R, T, cam_in, cam_ex):
    from xml.etree.ElementTree import Element,ElementTree
    K_str = ''.join(str(e)+" " for e in K)
    T_str = ''.join(str(e)+" " for e in T)
    R_str = ''.join(str(e)+" " for e in R)

    content = [
    {
        'name':'M',
        'type':'in',
        'rows':'3',
        'cols':'3',
        'dt':'d',
        'data':K_str
    },
    {
        'name':'D',
        'type':'in',
        'rows':'5',
        'cols':'1',
        'dt':'d',
        'data':'0 0 0 0 0'
    },
    {
        'name':'R',
        'type':'ex',
        'rows':'3',
        'cols':'3',
        'dt':'d',
        'data':R_str
    },
    {
        'name':'T',
        'type':'ex',
        'rows':'3',
        'cols':'1',
        'dt':'d',
        'data':T_str
    }
    ]
    
    root_in = Element('opencv_storage')
    tree_in = ElementTree(root_in)
    
    root_ex = Element('opencv_storage')
    tree_ex = ElementTree(root_ex)
    
    for cont in content:
        for k,v in cont.items():
            if k == 'name':
                child0 = Element(v)
                child0.set("type_id","opencv-matrix")
                continue
            if v == "in":
                root_in.append(child0)
            elif v == 'ex':
                root_ex.append(child0)
            else:
                child00 = Element(k)
                child00.text = v
                child0.append(child00)
    
    indent(root_in,0)
    tree_in.write(cam_in, 'UTF-8')#, xml_declaration=True)
    
    indent(root_ex,0)
    tree_ex.write(cam_ex, 'UTF-8')#, xml_declaration=True)

    with open(cam_in, 'r') as f:
        in_strs = f.readlines()
    with open(cam_in, 'w') as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.writelines(in_strs)
    
    with open(cam_ex, 'r') as f:
        ex_strs = f.readlines()
    with open(cam_ex, 'w') as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.writelines(ex_strs)

def abspath2remotepath(root, abspath):
    root_path = Path(root)
    search_str = "animation_mit"
    real_path = PureWindowsPath(abspath[abspath.index(search_str) + len(search_str)+1:])
    correct_path = Path(real_path)
    real_path = root_path / real_path
    return str(real_path)

def getitem(idx):
    sample = defaultdict(list) # return value
    oneframe = grouping[idx]
    imgnames = oneframe['imgpath']
    bboxs = oneframe['detections']
    cameras = oneframe['cameras']
    keypoints_3d = oneframe['keypoints_3d']

    for i in range(len(imgnames)):
        imgname = imgnames[i]

        # if os.path.exists(imgname[0:-4]+".pkl"):
        #     with open(imgname[0:-4]+".pkl", 'rb') as f:
        #         preprocess = pickle.load(f)
        #     image = preprocess['image']
        #     camera = preprocess['camera']
        #     bbox = preprocess['detections']
        #     sample['image_shapes_before_resize'].append([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        # else:
        bbox = bboxs[i]
        camera = cameras[i]
        bbox = scale_bbox(bbox, scale_bbox_ratio)
        imgname = abspath2remotepath(root, imgname)
        oriimage = cv2.imread(imgname)

        if crop:
            image = crop_image(oriimage, bbox)
            camera.update_after_crop(bbox)

        if image_shape is not None:
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, image_shape)
            camera.update_after_resize(image_shape_before_resize, image_shape)
            sample['image_shapes_before_resize'].append(image_shape_before_resize)

        sample['images'].append(oriimage)
        sample['cameras'].append(camera)
        sample['proj_matrices'].append(camera.projection)
        sample['detections'].append(bbox + (1.0,))
        sample['imagepath'].append(imgname)
        sample['bbox'].append(bbox)

    # save sample's index
    sample['indexes'] = idx
    return sample


def mkdir_gtske(f):
    if not os.path.exists(f):
        os.mkdir(f)
    if not os.path.exists(f+"\\skeleton_body"):
        os.mkdir(f + "\\skeleton_body")
    return f+"\\skeleton_body\\skeleton.txt"

from pathlib import *
import pickle

# savepath = Path("E:\\LIYUWEI\\H3.6m\\learnable-triangulation-pytorch\\logs\\eval_human36m_alg_AlgebraicTriangulationNet@04122019_13-48-17\\checkpoints\\0000\\pred_2d.pkl") #eval
savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_07-31-39/checkpoints/0000/pred_2d.pkl") #train
# savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_07-31-39_synmit_train_pred_2d/checkpoints/0000/pred_2d.pkl")
# savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_07-33-25/checkpoints/0000/pred_2d.pkl") #eval
# savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_07-33-25_synmit_val_pred_2d/checkpoints/0000/pred_2d.pkl")
# savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_09-06-00/checkpoints/0000/pred_2d.pkl") #test
# savepath = Path("logs/eval_synmit_alg_AlgebraicTriangulationNet@12122019_09-06-00_synmit_test_pred_2d/checkpoints/0000/pred_2d.pkl")
subset = "train"
print(savepath)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
root = "/data/liyuwei/Pose/dataset/animation_mit/"
# root = "E:\\LIYUWEI\\H3.6m\\animation_mit\\"
ignore_cameras =[]
scale_bbox_ratio = 1.0
crop = True
image_shape = [256, 256]

gt_db = []
pkl_name = os.path.join(root, 'list', 'pkl', subset + '_ltn.pkl')
########################### load from pkl
if os.path.exists(pkl_name):
    print("Loading dataset from: " + pkl_name)
    with open(pkl_name, 'rb') as f:
        grouping = pickle.load(f)

with savepath.open("rb") as fs:
    gt_db = pickle.load(fs)

index = gt_db['sample_indexes']
preds = gt_db['pred']

for i in range(len(index)):
    for bn in range(len(index[i])):
        inx = index[i][bn]
        pred = preds[i][bn].cpu().detach()
        sample = getitem(inx)

        # v1img = sample['images'][2]
        # joint_count = pred.shape[1]
        # pred[:,:,0:2] = pred[:,:,0:2] * (4)
        # for j in range(joint_count):
        #     cv2.circle(v1img, (int(pred[2,j,0]), int(pred[2,j,1])),5,(0,255,0),thickness=-1)
        # plt.figure()
        # plt.imshow(v1img)
        # plt.show()

        img_frame = Path(sample['imagepath'][0])

        save_keypoint_path = img_frame.parent / "openpose/just_input_fix-ltn17.txt"
        # print(save_keypoint_path)

        if not os.path.exists(save_keypoint_path.parent):
            os.mkdir(save_keypoint_path.parent)
        view_count = pred.shape[0]
        joint_count = pred.shape[1]
        pred[:,:,0:2] = pred[:,:,0:2] * 4

        for v in range(view_count):
            bbx = sample['bbox'][v]
            image_shape_before_resize = sample['image_shapes_before_resize'][v]
            pred[v,:,0] = pred[v,:,0] * image_shape_before_resize[0] / image_shape[0]
            pred[v,:,1] = pred[v,:,1] * image_shape_before_resize[1] / image_shape[1]
            pred[v,:,0] = pred[v,:,0] + bbx[0]
            pred[v,:,1] = pred[v,:,1] + bbx[1]

        camids = []
        for vi in range(view_count):
            camid = int(sample['imagepath'][vi][sample['imagepath'][vi].index("cam")+3:sample['imagepath'][vi].index("cam")+5])
            camids.append(camid)
        order = np.argsort(camids)

        sample['images'] = np.array(sample['images'])[order]
        sample['imagepath'] = np.array(sample['imagepath'])[order]
        pred = pred[order,:,:]

        # for vi in range(view_count):
        #     v1img = sample['images'][vi]
        #     joint_count = pred.shape[1]
        #     for j in range(joint_count):
        #         cv2.circle(v1img, (int(pred[vi,j,0]), int(pred[vi,j,1])),5,(0,255,0),thickness=-1)
        #     plt.figure()
        #     plt.title(sample['imagepath'][vi])
        #     plt.imshow(v1img)
        #     plt.show()

        input_fix = np.zeros((view_count*joint_count, 4))
        for j in range(joint_count):
            input_fix[j*view_count:j*view_count+view_count,0:2] = pred[:,j,0:2]
            input_fix[j*view_count:j*view_count+view_count,3] = pred[:,j,2]
        np.savetxt(save_keypoint_path, input_fix)

        print(i, view_count, save_keypoint_path)

        # ### save calibration folder
        # camfolder_root = os.path.join(h36m_root, sample['subject'], sample['action'],"OURS","calib")
        # # print(sample['subject'], sample['action'])
        # if not os.path.exists(camfolder_root):
        #     os.mkdir(camfolder_root)
        # else:
        #     if len(os.listdir(camfolder_root)) > 3:
        #         continue

        # idmapf_str = []
        # cam_count = len(sample['cameras'])
        # for camid in range(cam_count):
        #     # imgshape = sample['image_shapes_before_resize'][camid]
        #     imgshape = image_shape
        #     camname = "cam" + str(sample['camera_name'][camid]) + "_000000"
        #     camfolder = os.path.join(camfolder_root, camname)
        #     if not os.path.exists(camfolder):
        #         os.mkdir(camfolder)
        #     K = sample['cameras'][camid].K.reshape(-1)
        #     R = sample['cameras'][camid].R.reshape(-1)
        #     T = sample['cameras'][camid].t.reshape(-1)
        #     # print(R)
        #     cam_in = camfolder + "//intrinsic.xml"
        #     cam_ex = camfolder + "//extrinsics.xml"        
        #     write_cam(K, R, T, cam_in, cam_ex)  
        #     idmapf_str_cam = str(camid+1) + " " + "\"" + camname+"\"" + " " + str(imgshape[1]) + " " + str(imgshape[0])+"\n"
        #     idmapf_str.append(idmapf_str_cam)
        
        # f = open(camfolder_root+"\\Idmap.txt",'w')
        # f.write(str(cam_count) + " " + str(cam_count) + " \"Resolution\"\n")
        # f.writelines(idmapf_str)
        # f.close()

        # print(camfolder_root)

# fout.close()