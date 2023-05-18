"""Augment the original dataset with GenAug:
"""
import PIL
import os
import re
import pickle
import numpy as np
import hydra
import open3d as o3d
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from genaug_prompt import GenAugPrompt
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionDepth2ImgPipeline
import torch
import random
import cv2
import requests
from PIL import Image
import pybullet as p
from copy import deepcopy
import time
import logging
logging.getLogger("pybullet").setLevel(logging.ERROR)
from PIL import Image
import torch
from tqdm.auto import tqdm
from utils import get_pc, render_camera, reset_scene, create_obj, create_scene, visualize_rgb_pc
import glob
import cv2
import torch
import urllib.request
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import time
import copy
import sys
import diffusers
current_path = os.path.dirname(os.path.realpath(__file__))
class GenAug():
    """A simple image dataset class."""

    def __init__(self, rgb=None, depth=None, masks=None, camera=None, init_table=None, init_depth=None, init_table_mask=None):
        """A augmented RGB-D image dataset."""
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.safety_checker = lambda images, clip_input: (images, False)
        self.n_episodes = 0
        self.seed = 0
        physicsClient = p.connect(p.DIRECT)
        self.p = p
        self.model_base = current_path+ "/meshes"

        # stable diffusion depth2image
        self.depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.obj_bbox = []
        self.rgb = rgb
        self.depth=depth
        self.masks=masks
        self.camera = camera
        self.init_rgb, self.init_table_mask, self.init_depth = init_table, init_table_mask, init_depth
        self.zone =[[0, 0.25, -0.5, -0.3], [0.7, 1, -0.5, -0.3], [0.0, 0.25, 0.25, 0.4],
         [0.55, 0.8, 0.25, 0.4]]
        self.num_distractors = 3

    def obj2action(self, image, original_img, original_mask, target_name, dist_th):
        image = np.array(image)
        image_prev = np.array(image).copy()
        mask = np.array(original_mask)

        if len(np.where(mask == 255)[0])==0:
            return image_prev, False, '', None

        zoomed_rgb, zoomed_mask, current_zoomed_rgb, bounded = self.aug_preprocess_depth2img(original_img, mask, image, dist_th)

        if zoomed_rgb is None:
            return image_prev, False, '', None

        lang_prompt = "a {} on a table".format(target_name)
        with torch.cuda.amp.autocast(True):
            n_propmt = "bad, deformed, ugly, bad anotomy, low resolution"
            generated_image = self.depth2img_pipe(prompt=lang_prompt, image=zoomed_rgb, negative_prompt=n_propmt, strength=0.9).images[0]
            generated_image = np.array(generated_image)
            generated_image[np.array(zoomed_mask)==0] = 0
            generated_image = PIL.Image.fromarray(generated_image)

        black_image = np.zeros_like(image)
        generated_image = np.array(generated_image.resize((bounded[3] - bounded[2], bounded[1] - bounded[0])))
        black_image[bounded[0]:bounded[1], bounded[2]:bounded[3]] = generated_image
        black_image = np.array(PIL.Image.fromarray(black_image).resize((mask.shape[1], mask.shape[0])))
        image = np.array(PIL.Image.fromarray(image).resize((mask.shape[1], mask.shape[0])))
        image[mask == 255] = black_image[mask == 255]


        return image, True, target_name, mask

    def aug_room(self, image, mask, obj_name):
        image = PIL.Image.fromarray(image).resize((512, 512))
        mask = PIL.Image.fromarray(mask).resize((512, 512))
        image,  has_changed = self.get_room(image, mask, obj_name)
        return image

    def get_room(self, image, mask, new_name):
        image = np.array(image)
        zoomed_mask = PIL.Image.fromarray(np.array(mask))
        zoomed_rgb = PIL.Image.fromarray(image)
        lang_prompt = new_name

        with torch.cuda.amp.autocast(True):
            generated_image = self.pipe(prompt=lang_prompt, image=zoomed_rgb, mask_image=zoomed_mask).images[0]

        image = generated_image
        return np.array(image),True

    def get_table(self, table_prompt, camera_config):
        table_texture = random.choice(
            glob.glob("meshes/table_texture/*"))

        texture_id = p.loadTexture(table_texture)
        table_mode = random.choice(['wall', 'round_table'])
        rotation = Rot.from_rotvec([1.57, 0, 0]).as_matrix()
        rotated_quat = Rot.from_matrix(
            Rot.from_rotvec([0, 0, np.random.uniform(-0.25, 0.25)]).as_matrix() @ rotation).as_quat()
        table_path = "meshes/wall/{}.obj".format(table_mode)
        table_id = create_obj(self.p, table_path, (random.uniform(0.6, 0.8), 0.001, random.uniform(0.8, 1)), 0,
                                    [0.65, 0, 0.001], rotated_quat)
        self.p.changeVisualShape(table_id, -1, rgbaColor=[1,1,1, 1])
        self.p.changeVisualShape(table_id, -1, textureUniqueId=texture_id)

        init_color, init_depth, segm = render_camera(self.p, camera_config)
        init_mask = np.zeros_like(segm)
        init_mask[segm == table_id] = 255

        init_table = init_color.copy()
        init_table[init_mask == 0] = 255
        init_depth[init_mask == 0] = -1
        init_table = PIL.Image.fromarray(init_table).resize((512, 512))

        n_propmt = "bad, deformed, ugly, bad anotomy, low resolution"
        image = \
        self.depth2img_pipe(prompt=table_prompt, image=init_table, negative_prompt=n_propmt, strength=0.9).images[0]

        image = image.resize((init_mask.shape[1], init_mask.shape[0]))
        image = np.array(image)

        image[init_mask==0] = 255

        zone_mask = init_mask
        return image, zone_mask, np.array(init_depth)

    def swap_texture(self, image, original_img, obj_mask, original_name, lang_prompt):
        image = PIL.Image.fromarray(image).resize((512, 512))
        image, has_changed, obj_name, current_mask = self.obj2aug_depth(image, original_img, obj_mask, original_name, lang_prompt)

        return image, current_mask

    def obj2aug_depth(self, image, original_img, obj_mask, obj_name, lang_prompt):
        image = np.array(image)
        dist_th = np.random.randint(20, 30)
        zoomed_rgb, zoomed_mask, current_zoomed_rgb, bounded = self.aug_preprocess_depth2img(original_img, obj_mask, image, dist_th)
        if zoomed_rgb is None:
            return image, False, obj_name, None

        with torch.cuda.amp.autocast(True):
            n_propmt = "bad, deformed, ugly, bad anotomy, low resolution"

            generated_image = self.depth2img_pipe(prompt=lang_prompt, image=zoomed_rgb, negative_prompt=n_propmt, strength=0.9).images[0]
            generated_image = np.array(generated_image)
            generated_image[np.array(zoomed_mask)==0] = 0
            generated_image = PIL.Image.fromarray(generated_image)

        black_image = np.zeros_like(image)

        generated_image = np.array(generated_image.resize((bounded[3] - bounded[2], bounded[1] - bounded[0])))
        black_image[bounded[0]:bounded[1], bounded[2]:bounded[3]] = generated_image
        black_image = np.array(PIL.Image.fromarray(black_image).resize((obj_mask.shape[1], obj_mask.shape[0])))
        image = np.array(PIL.Image.fromarray(image).resize((obj_mask.shape[1], obj_mask.shape[0])))

        obj_mask = cv2.erode(obj_mask, np.ones((3,3), np.uint8))
        image[obj_mask==255] = black_image[obj_mask==255]
        return image, True, obj_name, obj_mask

    def aug_preprocess_depth2img(self, rgb, mask, current_rgb, dist_th):
        bounded_mask = np.zeros_like(rgb[:, :, :3])
        # get the bounding box
        x_min = max(0, np.min(np.where(mask == 255)[0])-dist_th)
        x_max = min(mask.shape[0], max(np.where(mask == 255)[0])+dist_th)
        y_min = max(0, np.min(np.where(mask == 255)[1])-dist_th)
        y_max = min(mask.shape[1], max(np.where(mask == 255)[1])+dist_th)

        zoomed_rgb = rgb[x_min:x_max, y_min:y_max]
        zoomed_mask = mask[x_min:x_max, y_min:y_max]
        bounded_mask[x_min:x_max, y_min: y_max] = 255
        current_zoomed_rgb = current_rgb[x_min:x_max, y_min:y_max]
        if current_zoomed_rgb.shape[0] < 10 or  current_zoomed_rgb.shape[1] < 10:
            return None, None, None, None

        zoomed_mask = PIL.Image.fromarray(zoomed_mask).resize((512, 512))
        bounded_mask = PIL.Image.fromarray(bounded_mask).resize((512, 512))
        zoomed_rgb = PIL.Image.fromarray(zoomed_rgb).resize((512, 512))

        current_zoomed_rgb = np.array(PIL.Image.fromarray(current_zoomed_rgb).resize((512, 512)))
        bounded_xmin = np.min(np.where(np.array(bounded_mask) == 255)[0])
        bounded_xmax = np.max(np.where(np.array(bounded_mask)== 255)[0])
        bounded_ymin = np.min(np.where(np.array(bounded_mask) == 255)[1])
        bounded_ymax = np.max(np.where(np.array(bounded_mask) == 255)[1])
        bounded = [bounded_xmin, bounded_xmax, bounded_ymin, bounded_ymax]

        return zoomed_rgb, zoomed_mask, current_zoomed_rgb, bounded

    def aug_preprocess(self, rgb, mask, dist_th, dilate_size):
        if dilate_size>0:
            mask = cv2.dilate(mask, np.ones((dilate_size,dilate_size), np.uint8))
        else:
            mask = cv2.erode(mask, np.ones((-dilate_size, -dilate_size), np.uint8))
        # if mask is less than a size, zoom in
        mask_w = max(np.where(mask == 255)[0]) - np.min(np.where(mask == 255)[0])
        mask_h = max(np.where(mask == 255)[1]) - np.min(np.where(mask == 255)[1])

        if mask_w<200 or mask_h<200: # zoom in
            bounded_mask = np.zeros_like(rgb[:, :, :3])
            #####  zoom in ####
            # get the bounding box
            x_min = max(0, np.min(np.where(mask == 255)[0])-dist_th)
            x_max = max(np.where(mask == 255)[0])+dist_th
            y_min = max(0, np.min(np.where(mask == 255)[1])-dist_th)
            y_max = max(np.where(mask == 255)[1])+dist_th

            zoomed_rgb = rgb[x_min:x_max, y_min:y_max]
            zoomed_mask = mask[x_min:x_max, y_min:y_max]
            bounded_mask[x_min:x_max, y_min: y_max] = 255
            # filling the hole to refine the mask
            contour, hier = cv2.findContours(zoomed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(zoomed_mask, [cnt], 0, 255, -1)
            zoomed_mask = cv2.cvtColor(zoomed_mask, cv2.COLOR_GRAY2BGR)
            # make mask thicker
            if dilate_size>0:
                dilate_th = np.random.randint(2, 10)
                zoomed_mask = cv2.dilate(zoomed_mask, np.ones((dilate_th, dilate_th), np.uint8))
            zoomed_mask = PIL.Image.fromarray(zoomed_mask).resize((512, 512))
            bounded_mask = PIL.Image.fromarray(bounded_mask).resize((512, 512))
            zoomed_rgb = PIL.Image.fromarray(zoomed_rgb).resize((512, 512))

            # get the bounding box
            bounded_xmin = np.min(np.where(np.array(bounded_mask) == 255)[0])
            bounded_xmax = np.max(np.where(np.array(bounded_mask)== 255)[0])
            bounded_ymin = np.min(np.where(np.array(bounded_mask) == 255)[1])
            bounded_ymax = np.max(np.where(np.array(bounded_mask) == 255)[1])
            bounded = [bounded_xmin, bounded_xmax, bounded_ymin, bounded_ymax]
        else:
            zoomed_mask = PIL.Image.fromarray(mask)
            zoomed_rgb = PIL.Image.fromarray(rgb)
            bounded = [0,512,0,512]

        return zoomed_rgb, zoomed_mask, bounded


    def action_augmentation(self, rgb_table, depth_table, depth, mask, action_type, camera_config, existing_obj):
        # first, fit the shape template to the pointcloud
        intrinsics, extrinsics = camera_config['intrinsics'], camera_config['extrinsics']
        name_list = os.listdir(self.model_base + '/{}'.format(action_type))
        for each_repeated_name in existing_obj:
            if each_repeated_name in name_list:
                name_list.remove(each_repeated_name)
        target_name = random.choice(name_list)

        if action_type == "distractor":
            obj_zones = self.zone
            zone_obj = random.choice(obj_zones)
            center = np.array([np.random.uniform(zone_obj[0], zone_obj[1]), np.random.uniform(zone_obj[2], zone_obj[3]),
                               np.random.uniform(0.05, 0.1)])
            scale = (np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1))
        else:
            existing_obj.append(target_name)
            obj_points = get_pc(depth, mask, intrinsics, extrinsics)
            center = np.mean(obj_points, 0)  # + delta_translation
            center[2] = 0.05
            scale_factor = np.random.uniform(1.0, 1.2)
            scale = (scale_factor, scale_factor, np.random.uniform(0.8, 1.2))

        model_path = self.model_base + "/{0}/{1}".format(action_type, target_name)

        new_color, new_depth, mask, obj_id = create_scene(self.p, model_path, center, scale, camera_config, action_type, self.obj_bbox)
        if mask is None:
            return None, None, new_depth, existing_obj

        if action_type != "distractor":
            min_x = self.p.getAABB(obj_id)[0][0]
            min_y = self.p.getAABB(obj_id)[0][1]
            max_x = self.p.getAABB(obj_id)[1][0]
            max_y = self.p.getAABB(obj_id)[1][1]
            self.obj_bbox.append(((min_x, min_y), (max_x, max_y)))
        rgb_table[mask == 255] = new_color[mask == 255]
        depth_table[mask == 255] = new_depth[mask == 255]
        rgb_table1 = PIL.Image.fromarray(rgb_table).resize((512, 512))

        # assuming all the rgb and mask are resized to 512x512
        dist_th = 50
        image, has_changed, obj_name, current_mask = self.obj2action(rgb_table1, rgb_table, mask, target_name, dist_th)

        return image, current_mask, depth_table, existing_obj

    def rgb_only_augmentation(self, texture=True, object=False, distractors=True, background=True, table=True, obj_names=None):
        # if image-only, global augmentation:
        prompt = GenAugPrompt()
        reset_scene(genaug.p)
        self.rgb = np.array(self.rgb)[:, :, :3]

        init_w, init_h = np.array(self.rgb).shape[0], np.array(self.rgb).shape[1]
        print('Augmenting without depth. To augment depth, please provide depth information when initialize GenAug: GenAug(rgb=image, depth=depth)')
        foreground_image = PIL.Image.fromarray(np.array(self.rgb.copy())[:, :, :3]).resize((512, 512))
        n_propmt = "bad, deformed, ugly, bad anotomy, low resolution"
        if obj_names is None:
            raise ValueError(f"Please provide object names in the format of ['object1', 'object2'...]")
        if self.masks is None:
            lang = "a table-top view of {}".format(obj_names)
            current_rgb = \
            self.depth2img_pipe(prompt=lang, image=foreground_image,
                                negative_prompt=n_propmt, strength=0.9).images[0]

            new_rgb = np.array(current_rgb.resize((init_h, init_w)))
        else:
            new_rgb = self.rgb.copy()
            for obj_i, obj_mask in enumerate(masks):
                if texture:
                    # swap texture of the demo objects
                    texture_prompt = "a " + prompt.material_name() + " " + obj_names[obj_i]
                    new_rgb, current_mask = genaug.swap_texture(new_rgb, new_rgb, obj_mask, obj_names[obj_i],
                                                                texture_prompt)

                else:
                    print(new_rgb.shape, obj_mask.shape, np.array(self.rgb).shape)
                    new_rgb[obj_mask == 255] = np.array(self.rgb)[obj_mask == 255]

            new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))

            #################### augment table #######################
            if table:
                current_rgb = new_rgb.copy()
                # # get table mask:
                combined_mask = np.logical_or.reduce(masks).astype(np.uint8) * 255
                new_rgb = PIL.Image.fromarray(new_rgb).resize((512, 512))
                new_rgb = self.depth2img_pipe(prompt="a table-top view of bunch of objects".format(obj_names), image=new_rgb,
                                    negative_prompt=n_propmt, strength=0.9).images[0]
                new_rgb = np.array(new_rgb.resize((init_h, init_w)))
                new_rgb[combined_mask== 255] = np.array(current_rgb)[combined_mask== 255]
            else:
                if self.init_rgb is None:
                    raise ValueError(f"Please provide rgbd and mask of the initial table")
                new_rgb, table_mask, new_depth = self.init_rgb, self.init_table_mask, self.init_depth
            new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))

        return new_rgb, None

    def rgb_depth_augmentation(self, texture=True, object=False, distractors=True, background=True, table=True, obj_names=None):
        # load the original dataset: images, masks, actions, and goal_conditions.
        prompt = GenAugPrompt()
        reset_scene(genaug.p)
        self.rgb = np.array(self.rgb)
        original_img = self.rgb.copy()
        depth = self.depth

        init_w, init_h = np.array(self.rgb).shape[0], np.array(self.rgb).shape[1]
        existing_obj = []

        if self.camera is None:
            raise ValueError(f"Please provide camera intrinsics and extrinsics to augment table")

        #################### augment table #######################
        if table:
            table_prompt = prompt.table_name()
            new_rgb, table_mask, new_depth = genaug.get_table(table_prompt, self.camera)
        else:
            if self.init_rgb is None or  self.init_table_mask is None or self.init_depth is None:
                raise ValueError(f"Please provide rgbd and mask of the initial table")
            new_rgb, table_mask, new_depth = self.init_rgb, self.init_table_mask, self.init_depth

        new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))
        if object:
            if self.masks is None:
                raise ValueError(f"Please provide masks to augment objects")
        for obj_i, obj_mask in enumerate(self.masks):
            action_type = "objects"
            if object:
                new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))
                new_rgb, place_mask, new_depth, existing_obj = genaug.action_augmentation(new_rgb, new_depth, depth,
                                                                                          obj_mask, action_type,
                                                                                          self.camera,
                                                                                          existing_obj)
            else:
                if texture:
                    # swap texture of the demo objects
                    texture_prompt = "a " + prompt.material_name() + " " + obj_names[obj_i]
                    new_rgb, current_mask = genaug.swap_texture(new_rgb, original_img, obj_mask, obj_names[obj_i], texture_prompt)
                    new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))
                else:
                    new_rgb[obj_mask == 255] = original_img[obj_mask == 255]

                intrinsics, extrinsics = self.camera['intrinsics'], self.camera['extrinsics']
                obj_points = get_pc(depth, obj_mask, intrinsics, extrinsics)
                obj_points = obj_points[obj_points[:, 2] > 0]
                point_cloud_o3d = o3d.cuda.pybind.geometry.PointCloud()
                point_cloud_o3d.points = o3d.utility.Vector3dVector(obj_points)

                point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=0.01)
                point_cloud_filtered, ind = point_cloud_o3d.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
                obj_points = np.asarray(point_cloud_filtered.points)

                min_x = min(obj_points[:, 0])-0.02
                min_y = min(obj_points[:, 1])-0.02
                max_x = max(obj_points[:, 0])+0.02
                max_y = max(obj_points[:, 1])+0.02

                genaug.obj_bbox.append(((min_x, min_y), (max_x, max_y)))
                new_depth[obj_mask == 255] = depth[obj_mask == 255]

            new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))

            ####################### add distractors #########################
            if distractors:
                action_type = 'distractor'
                distractor_id = 0
                while distractor_id <= self.num_distractors:
                    new_rgb1, pick_mask, new_depth1, existing_obj = genaug.action_augmentation(new_rgb, new_depth, depth,
                                                                                               obj_mask, action_type,
                                                                                               self.camera,
                                                                                               existing_obj)
                    if new_rgb1 is not None:
                        distractor_id += 1
                        new_rgb = np.array(PIL.Image.fromarray(new_rgb1).resize((init_h, init_w)))
                        new_depth = new_depth1

        new_depth[(new_rgb[:, :, 0] == 255) & (new_rgb[:, :, 1] == 255) & (new_rgb[:, :, 2] == 255)] = -1

        #################### add background room ####################
        if background:
            room_mask = np.zeros_like(
                new_rgb[:, :, 0])  # np.array(PIL.Image.open(data_path+"/empty_table.png"))[:, :, 0]
            room_mask[(new_rgb[:, :, 0] == 255) & (new_rgb[:, :, 1] == 255) & (new_rgb[:, :, 2] == 255)] = 255
            room_mask = cv2.dilate(room_mask, np.ones((5, 5), np.uint8))
            obj_name = prompt.room_name()
            new_rgb = genaug.aug_room(new_rgb, room_mask, obj_name)
            new_rgb = np.array(PIL.Image.fromarray(new_rgb).resize((init_h, init_w)))
        else:
            room_mask = np.zeros_like(new_rgb[:, :, 0])
            room_mask[(new_rgb[:, :, 0] == 255) & (new_rgb[:, :, 1] == 255) & (new_rgb[:, :, 2] == 255)] = 255
            room_mask = cv2.dilate(room_mask, np.ones((5, 5), np.uint8))
            new_rgb[room_mask == 255] = self.init_rgb[room_mask == 255]

        return new_rgb, new_depth

        # testing script:
    def augmentation(self, texture=True, object=False, distractors=True, background=True, table=True, obj_names=None):
        # if image-only, global augmentation:
        if self.depth is None:
            new_rgb, new_depth = self.rgb_only_augmentation(texture=texture, object=object, distractors=distractors, background=background, table=table, obj_names=obj_names)
        else:
            new_rgb, new_depth = self.rgb_depth_augmentation(texture=texture, object=object, distractors=distractors, background=background, table=table, obj_names=obj_names)

        return new_rgb, new_depth

def prepare_mask(image):
    num_obj = len(glob.glob(current_path + '/data/masks/*'))
    if num_obj>0:
        masks = []
        for i in range(num_obj):
            mask = np.array(PIL.Image.open(current_path+'/data/masks/mask{}.png'.format(i+1)))
            masks.append(mask)
        return masks

    mask = None

    def draw_polygon(event, x, y, flags, params):
        nonlocal mask
        if event == cv2.EVENT_LBUTTONDOWN:
            params['points'].append((x, y))
        if event == cv2.EVENT_MBUTTONDOWN:
            if len(params['points']) > 2:
                mask = np.zeros(params["image"].shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(params['points'])], (255, 255, 255))
                params["mask"] = mask
                cv2.imshow("Mask", mask)
        if event == cv2.EVENT_MOUSEMOVE:
            if len(params['points']) > 0:
                image_copy = params["image"].copy()
                cv2.polylines(image_copy, [np.array(params['points'])], True, (0, 0, 255), 2)
                mask = np.zeros(params["image"].shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(params['points'])], (255, 255, 255))
                cv2.imshow("Image", image_copy)

    # Load the image
    points = []
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_polygon, {"image": image, "mask": None, 'points': points})
    masks = []
    while True:
        if len(points) > 0:
            image_copy = image.copy()
            cv2.polylines(image_copy, [np.array(points)], True, (0, 0, 255), 2)
            cv2.imshow("Image", image_copy)
        else:
            cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == 27:  # Exit on ESC key
            break
        elif key == 32:  # Reset on space key
            points = []
            cv2.setMouseCallback("Image", draw_polygon, {"image": image, "mask": None, 'points': points})
        elif key == 13:  # Save and reset on Enter key
            cv2.imshow("Mask", mask)
            masks.append(mask)
            os.makedirs(current_path+"/data/masks/", mode=0o755, exist_ok=True)
            PIL.Image.fromarray(mask).save(current_path+'/data/masks/mask{}.png'.format(len(masks)))
            cv2.setMouseCallback("Image", draw_polygon, {"image": image, "mask": None, 'points': points})
    cv2.destroyAllWindows()
    return masks


def camera_config(image_size, intrinsics, extrinsics):
    CONFIG = {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'position': extrinsics[:3, 3],
        'rotation':  Rot.from_matrix(extrinsics[:3, :3]).as_quat(),
        'zrange': (0.01, 10.),
        'noise': False
    }
    return CONFIG




if __name__ == '__main__':

    intrinsics = np.array([[386.72210693359375, 0, 323.0413513183594], [0, 386.17559814453125, 242.57159423828125], [0, 0, 1]])
    extrinsics = np.array([[-0.99873792, 0.00936523, 0.04934428, 0.54809884],
                         [-0.03053219, 0.66687277, -0.7445458, 0.71942106],
                         [-0.0398792, -0.74511275, -0.66574518, 0.4798353],
                           [0,0,0,1]])

    image = PIL.Image.open("data/label_0.png")
    image_size = (np.array(image).shape[0], np.array(image).shape[1])
    camera = camera_config(image_size, intrinsics, extrinsics)
    ######################### prepare the segmentation masks of objects ##########################
    # [left mouse]: add a new point | [right mouse]: finish | [space]: reset | [esc]: exit
    masks = prepare_mask(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    ##############################################################################################

    data = pickle.load(open('data/label_0.pkl', 'rb'))
    depth = data['depth'] / 1000

    init_color = np.load("data/empty_color.npy")
    init_depth = np.load("data/empty_depth.npy") / 1000
    init_mask = np.array(PIL.Image.open("data/empty_table.png"))[:, :, 0]
    init_mask[init_mask > 230] = 255
    init_mask[init_mask != 255] = 0
    init_color[init_mask != 255] = 255

    genaug = GenAug(rgb=image, depth=depth, masks=masks, camera=camera)

    # for num_aug in range(100):
    new_rgb, new_depth = genaug.augmentation(obj_names=['a small paper box', 'basket'], object=False, distractors=False) #default: texture=True, object=False, distractors=True, background=True, table=True
    # PIL.Image.fromarray(new_rgb).save("/home/zoeyc/Downloads/outdoor2_{}aug.jpg".format(num_aug))
    PIL.Image.fromarray(new_rgb).show()

    visualize_rgb_pc(new_rgb, new_depth, camera['intrinsics'], camera['extrinsics'])
