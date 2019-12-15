#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import numpy as np
from collections import defaultdict
import cv2
from PIL import Image
import matplotlib.pyplot as plt


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


labels_path = "E:\\LIYUWEI\\H3.6m\\learnable-triangulation-pytorch\\data\\human36m\\extra\\human36m-multiview-labels-GTbboxes.npy"
h36m_root = "E:\\LIYUWEI\\H3.6m\\h36m-fetch\\processed\\"
labels = np.load(labels_path, allow_pickle=True).item()
test = False
train = True
with_damaged_actions = True
retain_every_n_frames_in_test = 1
pred_results_path = None 
ignore_cameras =[]
scale_bbox_ratio = 1.0
undistort_images = False
crop = True
image_shape = [384, 384]

n_cameras = len(labels['camera_names'])
train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
test_subjects = ['S9', 'S11']
# test_subjects = ['S11']
train_subjects = list(labels['subject_names'].index(x) for x in train_subjects)
test_subjects  = list(labels['subject_names'].index(x) for x in test_subjects)

indices = []

if train:
    mask = np.isin(labels['table']['subject_idx'], train_subjects, assume_unique=True)
    indices.append(np.nonzero(mask)[0])
if test:
    mask = np.isin(labels['table']['subject_idx'], test_subjects, assume_unique=True)

    if not with_damaged_actions:
        mask_S9 = labels['table']['subject_idx'] == labels['subject_names'].index('S9')

        damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
        damaged_actions = [labels['action_names'].index(x) for x in damaged_actions]
        mask_damaged_actions = np.isin(labels['table']['action_idx'], damaged_actions)

        mask &= ~(mask_S9 & mask_damaged_actions)

    indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

labels['table'] = labels['table'][np.concatenate(indices)]
print(len(labels['table']))
num_keypoints = 17
assert labels['table']['keypoints'].shape[1] == 17, "Use a newer 'labels' file"

pred_results = None
if pred_results_path is not None:
    pred_results = np.load(pred_results_path, allow_pickle=True)
    keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
    keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]
    assert len(self.keypoints_3d_pred) == len(self)

def getitem(idx):
    sample = defaultdict(list) # return value
    shot = labels['table'][idx]

    subject = labels['subject_names'][shot['subject_idx']]
    action = labels['action_names'][shot['action_idx']]

    frame_idx = shot['frame_idx']

    for camera_idx, camera_name in enumerate(labels['camera_names']):
        if camera_idx in ignore_cameras:
            continue

        # # load bounding box
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
        print(bbox)
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        print(bbox_height, bbox_width)
        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            continue

        # # scale the bounding box
        # bbox = scale_bbox(bbox, scale_bbox_ratio)
        sample['bbox'].append(bbox)

        # load image
        image_path = os.path.join(
            h36m_root, subject, action, 'imageSequence' + '-undistorted' * undistort_images,
            camera_name, 'img_%06d.jpg' % (frame_idx+1))
        # assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        if not os.path.isfile(image_path):
            continue
        oriimg = cv2.imread(image_path)
        image = cv2.imread(image_path)

        # # load camera
        shot_camera = labels['cameras'][shot['subject_idx'], camera_idx]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

        if crop:
            # crop image
            image = crop_image(oriimg, bbox)
            retval_camera.update_after_crop(bbox)

        if image_shape is not None:
            # resize
            # image_shape_before_resize = image.shape[:2]
            image_shape_before_resize = [bbox[2] - bbox[0],bbox[3] - bbox[1]]
            # assert(image_shape_before_resize[0] == image_shape_before_resize_fbbox[0])
            # assert(image_shape_before_resize[1] == image_shape_before_resize_fbbox[1])
            # image = resize_image(image, image_shape)
            # retval_camera.update_after_resize(image_shape_before_resize, image_shape)

            sample['image_shapes_before_resize'].append(image_shape_before_resize)

        sample['imgpath'].append(image_path)
        sample['images'].append(oriimg)
        sample['images_crop'].append(image)

        sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
        sample['cameras'].append(retval_camera)
        sample['camera_name'].append(camera_name)
        sample['proj_matrices'].append(retval_camera.projection)

    # 3D keypoints
    # add dummy confidences
    sample['keypoints_3d'] = np.pad(
        shot['keypoints'][:num_keypoints],
        ((0,0), (0,1)), 'constant', constant_values=1.0)

    # sample['keypoints_3d'] = shot['keypoints'][:num_keypoints]
    sample['subject'] = subject
    sample['action'] = action
    sample['frame'] = frame_idx+1

    # build cuboid
    # base_point = sample['keypoints_3d'][6, :3]
    # sides = np.array([cuboid_side, cuboid_side, cuboid_side])
    # position = base_point - sides / 2
    # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

    # save sample's index
    sample['indexes'] = idx

    sample.default_factory = None
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
savepath = Path("E:\\LIYUWEI\\H3.6m\\learnable-triangulation-pytorch\\logs\\eval_human36m_alg_AlgebraicTriangulationNet@02122019_23-38-37\\checkpoints\\0000\\pred_2d.pkl") #train
print(savepath)
with savepath.open("rb") as fs:
    gt_db = pickle.load(fs)

index = gt_db['sample_indexes']
preds = gt_db['pred']
# fout = open("trainout.txt","w")
for i in range(len(index)):
    for bn in range(len(index[i])):
        inx = index[i][bn]
        pred = preds[i][bn].cpu().detach()
        sample = getitem(inx)

        v1img = sample['images_crop'][2]  
        joint_count = pred.shape[1]
        pred[:,:,0:2] = pred[:,:,0:2] * (384 / 96)
        for j in range(joint_count):
            cv2.circle(v1img, (int(pred[2,j,0]), int(pred[2,j,1])),5,(0,255,0),thickness=-1)      
        plt.figure()
        plt.imshow(v1img)
        plt.show()

'''  
        # print(i)
        sample = getitem(inx)
        if len(sample.keys())==0:
            continue
        # if sample['action'] != "Directions-2":
        #     continue

        ### save gt_skeleton
        # save_keypoint_path = os.path.join(h36m_root, sample['subject'], sample['action'],"OURS", str(sample['frame']), "gt_openpose")
        # print(save_keypoint_path)
        # fout.write(save_keypoint_path + "\n")
        # save_file_name = mkdir_gtske(save_keypoint_path)
        img_frame = int(sample['imgpath'][0][-10:-4])
        assert(img_frame == sample['frame'])
        # print(sample['subject'], sample['action'], img_frame)
        # np.savetxt(save_file_name, sample['keypoints_3d'], fmt="%8f %8f %8f")
        # print(str(i) + "/" + str(len(labels['table'])) + save_file_name)
        # print(sample['imgpath'])

        save_keypoint_path = os.path.join(h36m_root, sample['subject'], sample['action'],"OURS", str(sample['frame']), "openpose\\just_input_fix-undistorted.txt")
        print(save_keypoint_path)
        assert(os.path.exists(save_keypoint_path))
        continue
        # save_bbox_path = os.path.join(h36m_root, sample['subject'], sample['action'], "OURS", str(sample['frame']),"bbox")
        # if not os.path.exists(save_bbox_path):
        #     os.mkdir(save_bbox_path)
        # bboxs = sample['bbox']
        # camnames = sample['camera_name']
        # for cam in range(len(camnames)):
        #     np.savetxt(save_bbox_path + "\\" + camnames[cam] + "_bbx.txt", bboxs[cam])
        #     print(save_bbox_path + "\\" + camnames[cam] + "_bbx.txt")


        if not os.path.exists(save_keypoint_path):
            os.mkdir(save_keypoint_path)
        view_count = pred.shape[0]
        # if view_count == 3:
        # print(view_count)
        # if view_count == 3:
        #     real_view_count = 3
        #     view_count = 4
        #     print(save_keypoint_path)
        #     continue

        joint_count = pred.shape[1]
        pred[:,:,0:2] = pred[:,:,0:2] * (384 / 96)
        for v in range(view_count):
            bbx = sample['bbox'][v]
            image_shape_before_resize = sample['image_shapes_before_resize'][v]
            pred[v,:,0] = pred[v,:,0] * image_shape_before_resize[0] / image_shape[0]
            pred[v,:,1] = pred[v,:,1] * image_shape_before_resize[1] / image_shape[1]
            pred[v,:,0] = pred[v,:,0] + bbx[0]
            pred[v,:,1] = pred[v,:,1] + bbx[1]

        input_fix = np.zeros((view_count*joint_count, 4))
        for j in range(joint_count):
            input_fix[j*view_count:j*view_count+view_count,0:2] = pred[:,j,0:2]
            input_fix[j*view_count:j*view_count+view_count,3] = pred[:,j,2]
        np.savetxt(save_keypoint_path + "\\just_input_fix-undistorted.txt", input_fix)

        
        # v1img = sample['images'][0]
        # for j in range(joint_count):
        #     cv2.circle(v1img, (int(pred[0,j,0]), int(pred[0,j,1])),5,(0,255,0),thickness=-1)        
        # plt.figure()
        # plt.imshow(v1img)
        # plt.show()
        # if i % 100 == 0:
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

# fout.close()'''