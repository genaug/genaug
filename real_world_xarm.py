import numpy as np
import pyrealsense2 as rs
from time import sleep
import pybullet as p
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import os
import pickle
from scipy.spatial.transform import Rotation as Rot

LABELS_DIR = '../labels'
calib_position = np.array([0.5380988422830156, 0.7284210596582703, 0.46183529911096044])
calib_rotation = [0.004902906080955535, 0.9062262913069816, -0.41991621434495513, 0.04899227884224285]

CAMERA_CONFIG = {
    'intrinsics': [386.72210693359375, 0.0, 323.0413513183594, 0.0, 386.17559814453125, 242.57159423828125, 0.0, 0.0, 1.0],
    'position': calib_position,
    'rotation': calib_rotation,
}
XARM_SDK = '../../../xArm-Python-SDK'
XARM_IP = '192.168.1.220'
GRIPPER_HEIGHT = 0.11

import sys

sys.path.append(XARM_SDK)
from xarm.wrapper import XArmAPI


def take_photo():
    """Take photo using RealSense camera."""
    # configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # align depth
    align = rs.align(rs.stream.color)

    # start streaming
    pipeline.start(config)

    # sleep necessary to ensure camera adjusts to the lighting
    sleep(2)

    # get color and depth frames
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # convert frames to numpy arrays
    color = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
    depth = np.asanyarray(depth_frame.get_data(), dtype=np.float32)

    # stop streaming
    pipeline.stop()

    return color, depth


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image."""
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud."""
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, 'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud."""
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    heightmap[py, px] = points[:, 2]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def reconstruct_heightmaps(color, depth, config, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    intrinsics = np.array(config['intrinsics']).reshape(3, 3)
    xyz = get_pointcloud(depth, intrinsics) / 1000
    position = np.array(config['position']).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotation = np.array(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    xyz = transform_pointcloud(xyz, transform)
    heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
    return heightmap, colormap


def mouse_to_xyz(event, colormap, heightmap, bounds):
    """Converts mouse click event to an (x, y, z) coordinate in the real world."""
    click_x = colormap.shape[1] - event.x
    click_y = event.y
    x = click_x / colormap.shape[1] * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    y = click_y / colormap.shape[0] * (bounds[1, 1] - bounds[1, 0]) + bounds[1, 0]
    z = 0
    if 0 <= click_y < heightmap.shape[0] and 0 <= click_x < heightmap.shape[1]:
        z = heightmap[click_y, click_x]
    if z == 0:
        total = 0
        count = 0
        for i in range(click_y - 1, click_y + 1):
            for j in range(click_x - 1, click_x + 1):
                if 0 <= i < colormap.shape[0] and 0 <= j < colormap.shape[1] and heightmap[i, j] != 0:
                    total += heightmap[i, j]
                    count += 1
        if count > 0:
            z = total / count
    return x, y, z


def save_label(color, depth, colormap, heightmap, pick_pos, place_pos, scene_name):
    data = {
        'color': color,
        'depth': depth,
        'colormap': colormap,
        'heightmap': heightmap,
        'pick_pos': pick_pos,
        'place_pos': place_pos,
    }
    path = os.path.join(LABELS_DIR, scene_name)
    if not os.path.exists(path):
        os.makedirs(path)
    label_id = len(os.listdir(path))
    path = os.path.join(path, f'label_{label_id}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print('saved to ' + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_name', type=str)
    arg = parser.parse_args()

    arm = XArmAPI(XARM_IP)
    arm.set_collision_sensitivity(1)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    while True:
        input('hit enter to take a new photo...')
        color, depth = take_photo()

        bounds = np.array([[0, 1], [-0.5, 0.5], [-0.5, 0.5]])
        pixel_size = 0.003125
        heightmap, colormap = reconstruct_heightmaps(color, depth, CAMERA_CONFIG, bounds, pixel_size)

        root = tk.Tk()
        canvas = tk.Canvas(root, width=colormap.shape[1], height=colormap.shape[0])
        canvas.pack()

        image = Image.fromarray(np.flip(colormap, axis=1))
        image_tk = ImageTk.PhotoImage(image)
        canvas.create_image(colormap.shape[1] / 2, colormap.shape[0] / 2, image=image_tk)

        pick_pos = (0, 0, 0)
        place_pos = (0, 0, 0)

        def left_click(event):
            nonlocal canvas, pick_pos, colormap, heightmap, bounds
            canvas.delete('pick')
            canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='#0000ff', width=0, tag='pick')
            pick_pos = mouse_to_xyz(event, colormap, heightmap, bounds)
            print('pick pos:', pick_pos)

        def right_click(event):
            nonlocal canvas, place_pos, colormap, heightmap, bounds
            canvas.delete('place')
            canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='#00ff00', width=0, tag='place')
            place_pos = mouse_to_xyz(event, colormap, heightmap, bounds)
            print('place pos:', place_pos)

        canvas.bind('<Button-1>', left_click)
        canvas.bind('<Button-3>', right_click)
        root.mainloop()

        pick_x = 1000 * pick_pos[0]
        pick_y = 1000 * pick_pos[1]
        pick_z = 1000 * (pick_pos[2] + GRIPPER_HEIGHT)
        arm.set_position(x=pick_x, y=pick_y, z=pick_z + 200, roll=-180, pitch=0, yaw=0, speed=200, wait=True)
        arm.set_vacuum_gripper(True)
        arm.set_position(x=pick_x, y=pick_y, z=pick_z, roll=-180, pitch=0, yaw=0, speed=100, wait=True)
        sleep(1)
        arm.set_position(x=pick_x, y=pick_y, z=pick_z + 200, roll=-180, pitch=0, yaw=0, speed=100, wait=True)

        place_x = 1000 * place_pos[0]
        place_y = 1000 * place_pos[1]
        place_z = 1000 * (pick_pos[2] + place_pos[2] + GRIPPER_HEIGHT)
        arm.set_position(x=place_x, y=place_y, z=place_z + 200, roll=-180, pitch=0, yaw=0, speed=200, wait=True)
        arm.set_position(x=place_x, y=place_y, z=place_z + 20, roll=-180, pitch=0, yaw=0, speed=100, wait=True)
        arm.set_vacuum_gripper(False)
        sleep(1)
        arm.set_position(x=place_x, y=place_y, z=place_z + 200, roll=-180, pitch=0, yaw=0, speed=100, wait=True)

        arm.set_position(x=100, y=0, z=600, roll=-180, pitch=0, yaw=0, speed=200, wait=True)

        key = input('save? (y/n/q): ')
        if key == 'q':
            break
        elif key == 'y':
            save_label(color, depth, colormap, heightmap, pick_pos, place_pos, arg.scene_name)

    arm.disconnect()


if __name__ == '__main__':
    main()