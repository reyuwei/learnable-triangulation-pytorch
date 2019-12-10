import os
from collections import defaultdict
import pickle

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils import volumetric
import xml.etree.cElementTree as ET
import json_tricks as json
import re
import matplotlib.pyplot as plt
from pathlib import *

class SynMITMultiviewDataset(Dataset):
    """
        Human3.6M for multiview tasks.
    """
    def __init__(self,
                 synmit_root='/Vol1/dbstore/datasets/Human3.6M/processed/',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 crop=True,
                 norm_image=True,
                 cuboid_side=2,
                 scale_bbox=1.5,
                 view_count=8,
                 joint_count=17,
                 kind="synmit",
                 ):

        self.dataset = kind
        if train == False and test == True:
            self.is_train = False
            self.subset = "val"
        else:
            self.is_train = True
            self.subset = "train"

        self.scale_bbox = scale_bbox
        self.crop = crop
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.root = synmit_root
        self.image_shape = image_shape
        self.db = []

        self.view_count = view_count
        self.ransac_min_weight = 1e-5
        self.joint_count = joint_count
        self.grouping = self._get_db()
        self.group_size = len(self.grouping)

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred
            assert len(self.keypoints_3d_pred) == len(self)

        print("dataset length: " + str(len(self)))

    def abspath2remotepath(self, abspath):
        root_path = Path(self.root)
        search_str = "animation_mit"
        real_path = PureWindowsPath(abspath[abspath.index(search_str) + len(search_str)+1:])
        correct_path = Path(real_path)
        real_path = root_path / real_path
        return str(real_path)

    def cam2projm(self, camera):
        K = camera['K']
        R = camera['R']
        T = camera['T']
        RT = np.hstack([R,T])
        return np.matmul(K,RT)

    def parsecalibration(self, calibfolder):

        extrinsics = os.path.join(calibfolder, 'extrinsics.xml')
        intrinsic = os.path.join(calibfolder, 'intrinsic.xml')

        tree = ET.ElementTree(file=extrinsics)
        root = tree.getroot()
        # get R
        R_node = root[0]
        text = R_node[3].text
        R_data = re.split('[\s]\s*', text)
        R_data.remove('')
        R = list(map(eval, R_data))
        R = np.array(R)
        R = R.reshape(3, 3)
        # print(R)

        # get T
        T_node = root[1]
        text = T_node[3].text
        T_data = re.split('[\s]\s*', text)
        T_data.remove('')
        T = list(map(eval, T_data))
        T = np.array(T)
        T = T.reshape(3, 1)

        # load intrinsic
        tree = ET.ElementTree(file=intrinsic)
        root = tree.getroot()
        # get K
        date = root[0]
        if date.tag == "date":
            M_node = 2
        else:
            M_node = 0

        K_node = root[M_node]
        # K_node = root[2]
        text = K_node[3].text
        K_data = re.split('[\s]\s*', text)
        K_data.remove('')
        K = list(map(eval, K_data))
        K = np.array(K)
        K = K.reshape(3, 3)

        return K, R, T

    def openpose25toh36m17(selfs, joints_3d):
        mapid = [11,10,9,12,13,14,8,1,1,16,4,3,2,5,6,7,0]
        joints_3d_17 = joints_3d[mapid]
        joints_3d_17[7] = (joints_3d[1] + joints_3d[8])/2
        joints_3d_17[9] = (joints_3d[15] + joints_3d[16]) / 2
        return joints_3d_17

    def project3d(self, K, R, T, joints_3d):
        joint_count = joints_3d.shape[0]

        RT = np.c_[R, T]
        joints_3d_hom = np.c_[joints_3d, np.ones(joint_count)]

        joints_3d_w2c = np.matmul(RT, joints_3d_hom.transpose())
        joint_2d_wh = np.matmul(K, joints_3d_w2c)
        joint_2d_w = joint_2d_wh / joint_2d_wh[2]
        joint_2d_w = joint_2d_w[0:2,:].transpose()

        # joints_2d_vis_x = np.array(joint_2d_w[:,0] >= 0) & np.array(joint_2d_w[:,0] < cols)
        # joints_2d_vis_y = np.array(joint_2d_w[:,1] >= 0) & np.array(joint_2d_w[:,1] < rows)
        # joints_2d_vis = (joints_2d_vis_x.flatten() & joints_2d_vis_y.flatten()).astype(int).flatten()

        return joint_2d_w

    def _get_db(self):
        gt_db = []
        pkl_name = os.path.join(self.root, 'list', 'pkl', self.subset + '_ltn.pkl')
        ########################### load from pkl
        if os.path.exists(pkl_name):
            print("Loading dataset from: " + pkl_name)
            with open(pkl_name, 'rb') as f:
                gt_db = pickle.load(f)
            return gt_db
        else:
            if not os.path.exists(os.path.join(self.root, 'list', 'pkl')):
                os.mkdir(os.path.join(self.root, 'list', 'pkl'))

        ############################ load from folder
        file_name = os.path.join(self.root, 'list', self.subset + 'lst.txt')
        json_name = os.path.join(self.root, 'list', 'datalist.json')
        with open(json_name, 'r') as load_f:
            load_dict = json.load(load_f)
        with open(file_name) as anno_file:
            anno = anno_file.readlines()

        print("Total frames: " + str(len(anno)))

        for items in anno:  # per frame
            if (len(gt_db)) % 100 == 0:
                print("Loading " + str(len(gt_db)) + " Frames")
            frame_folder = items.split(";")[0]
            frame_folder_abs = self.abspath2remotepath(frame_folder)
            network_pred_path = os.path.join(frame_folder_abs, 'openpose', 'network_skeleton_body')

            dict_id = int(items.split(";")[1])
            framefiles = os.listdir(frame_folder_abs)
            imgname_pattern = load_dict[dict_id]['image_name_format']

            ## load 3d ground truth
            joints_3d_file = os.path.join(frame_folder_abs, load_dict[dict_id]['gt_skeleton'],
                                          'skeleton_body', 'skeleton.txt')
            joints_3d_25 = np.loadtxt(joints_3d_file)
            joints_3d = self.openpose25toh36m17(joints_3d_25)

            imgnames = []
            cameras = []
            bbxs = []
            for imgfile in framefiles:
                if re.match(imgname_pattern, imgfile) is not None:
                    imgname = os.path.join(frame_folder_abs, imgfile)
                    imgnames.append(imgname)

                    camname_len = int(imgname_pattern[9])
                    camname = imgname[imgname.index("cam"):imgname.index("cam") + 3 + camname_len] + "_000000"
                    calibration_folder = self.abspath2remotepath(load_dict[dict_id]['calibration'])
                    calibration_folder_camera = os.path.join(calibration_folder, camname)

                    ## load camera
                    K, R, T = self.parsecalibration(calibration_folder_camera)
                    camera = Camera(R, T, K, name = camname)
                    cameras.append(camera)

                    ## project 3D to 2D and compute bounding box
                    joints_2d = self.project3d(K, R, T, joints_3d)
                    center = joints_2d[6]
                    height = np.max(joints_2d[:,1]) - np.min(joints_2d[:,1])
                    width = np.max(joints_2d[:,0]) - np.min(joints_2d[:,0])
                    bbox_size = max(height, width)
                    bbox_size = bbox_size * 1.5
                    half_bbox_size = int(bbox_size / 2)
                    bbox = (center[0] - half_bbox_size, center[1] - half_bbox_size,  center[0] + half_bbox_size, center[1] + half_bbox_size)
                    bbxs.append(bbox)
                    # left upper right lower

                    # image = cv2.imread(imgname)
                    # image = crop_image(image, bbox)
                    # camera.update_after_crop(bbox)
                    # image_shape_before_resize = image.shape[:2]
                    #
                    # image = resize_image(image, self.image_shape)
                    # camera.update_after_resize(image_shape_before_resize, self.image_shape)
                    #
                    # joints_2d = self.project3d(camera.K, camera.R, camera.t, joints_3d)
                    # for j in joints_2d: # gt projection
                    #     cv2.circle(image, (int(j[0]), int(j[1])), 3, (0,255,0))
                    # plt.figure()
                    # plt.imshow(image)
                    # if not os.path.exists("vis"):
                    #     os.mkdir("vis")
                    # plt.savefig("vis\\" +imgfile)


            gt_db.append(
                {
                    'keypoints_3d': joints_3d,
                    'imgpath': imgnames,
                    'cameras': cameras,
                    'detections': bbxs,
                }
            )

        with open(pkl_name, 'wb') as f:
            pickle.dump(gt_db, f, pickle.HIGHEST_PROTOCOL)

        print("Init Dataset Done!!")
        return gt_db

    def __len__(self):
        return self.group_size

    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        oneframe = self.grouping[idx]
        imgnames = oneframe['imgpath']
        bboxs = oneframe['detections']
        cameras = oneframe['cameras']
        keypoints_3d = oneframe['keypoints_3d']

        for i in range(len(imgnames)):
            imgname = imgnames[i]
            bbox = bboxs[i]
            camera = cameras[i]

            bbox = scale_bbox(bbox, self.scale_bbox)
            image = cv2.imread(imgname)
            if self.crop:
                image = crop_image(image, bbox)
                camera.update_after_crop(bbox)

            if self.image_shape is not None:
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                camera.update_after_resize(image_shape_before_resize, self.image_shape)
                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            sample['cameras'].append(camera)
            sample['proj_matrices'].append(camera.projection)
            sample['detections'].append(bbox + (1.0,))

        # build cuboid
        sample['keypoints_3d'] = np.pad(
            keypoints_3d,
            ((0, 0), (0, 1)), 'constant', constant_values=1.0)
        base_point = sample['keypoints_3d'][6, :3]
        sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        position = base_point - sides / 2
        sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # save sample's index
        sample['indexes'] = idx

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        sample.default_factory = None
        return sample

    def evaluate_using_per_pose_error(self, per_pose_error, split_by_subject):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(1):
                action_mask = mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores['all'] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            for k, v in action_scores.items():
                action_scores[k] = v['total_loss'] / v['frame_count']

            return action_scores

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        for subject_idx in range(1):
            subject_mask = np.ones_like(per_pose_error, dtype=bool)
            subject_scores['all'] = evaluate_by_actions(self, per_pose_error, subject_mask)

        return subject_scores

    def evaluate(self, keypoints_3d_predicted):
        keypoints_gt = []
        for i in range(len(self)):
            keypoints_gt.append(self.grouping[0]['keypoints_3d'])
        keypoints_gt = np.array(keypoints_gt)

        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        # mean error per 16/17 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)

        # relative mean error per 16/17 joints in mm, for each pose
        root_index = 6
        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error, True),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative, True)
        }

        return result['per_pose_error_relative']['Average']['Average'], result
        return np.mean(per_pose_error), {}