import numpy as np
from scipy.spatial.transform import Rotation as Rot
import glob
import os
import time
import logging
import open3d as o3d
from copy import deepcopy
import copy
import time
logging.getLogger("pybullet").setLevel(logging.ERROR)
def get_pc( depth, mask, intrinsics, extrinsics):
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(extrinsics[i, :] * homogen_points, axis=-1)

    return np.array(points)[mask == 255].reshape(-1, 3)

def visualize_rgb_pc(rgb, depth, intrinsics, extrinsics, animation=True):
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(extrinsics[i, :] * homogen_points, axis=-1)

    pcd = o3d.geometry.PointCloud()
    pts = np.array(points).reshape(-1, 3)
    clr = np.array(rgb).reshape(-1, 3) / 255
    center = np.mean(pts, 0)
    keep_index = np.where(
        (pts[:, 0] < center[0] + 1.5) & (pts[:, 0] > center[0] - 1.5) & (pts[:, 1] < center[1] + 1.5) & (
                    pts[:, 1] > center[1] - 1.5) & (pts[:, 2] < 0.4))
    pcd.points = o3d.utility.Vector3dVector(pts[keep_index])
    pcd.colors = o3d.utility.Vector3dVector(clr[keep_index])
    if animation:
        custom_draw_geometry_with_rotation(pcd)
    else:
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    return pcd


def custom_draw_geometry_with_camera_trajectory(pcd, path):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "{}/camera_trajectory.json".format(path))
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )


    def move_forward(vis):

        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
                    np.asarray(depth), dpi = 1)
            plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
                    np.asarray(image), dpi = 1)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("../../TestData/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    def rotate_view(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index==-1:
            ctr.camera_local_translate(forward=0, right=0, up=0.1)
            ctr.set_zoom(0.5)

        ctr.rotate(5, 0)

        glb.index+=1
        if glb.index>500:
            vis.close()
        return False

    pcd1 = copy.deepcopy(pcd)
    R = pcd1.get_rotation_matrix_from_xyz((-np.pi/(2.1), 0, np.pi))
    pcd1 = pcd1.rotate(R, center=(0,0,0))
    o3d.visualization.draw_geometries_with_animation_callback([pcd1], rotate_view)

def render_camera(p, config, image_size=None, shadow=1):
    """Render RGB-D image with specified camera configuration."""
    if not image_size:
        image_size = config['image_size']

    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = config['position'] + lookdir

    focal_len = config['intrinsics'][0][0]
    znear, zfar = config['zrange']
    viewm = p.computeViewMatrix(config['position'], lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    aspect_ratio = image_size[1] / image_size[0]
    projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    _, _, color, depth, segm = p.getCameraImage(width=image_size[1], height=image_size[0], viewMatrix=viewm, projectionMatrix=projm, shadow=shadow, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]

    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    depth = (2. * znear * zfar) / depth

    segm = np.uint8(segm).reshape(depth_image_size)

    return color, depth, segm


def check_inside(point, bbox):
    x, y = point
    top_left, bottom_right = bbox
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    return top_x <= x <= bottom_x and top_y <= y <= bottom_y



def reset_scene(p):
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    rotation = Rot.from_rotvec([1.57, 0, 0]).as_matrix()
    rotated_quat = Rot.from_matrix(
        Rot.from_rotvec([0, 0, 0]).as_matrix() @ rotation).as_quat()
    workspace = create_obj(p, "/home/zoeyc/github/FunAug/cliport/cliport/environments/assets/wall/wall.obj",
                                (10, 0.001, 10), 0,
                                [0.5, 0, -0.005], rotated_quat)
    p.changeVisualShape(workspace, -1, rgbaColor=[0.5,0.5,0.5, 1])


def create_obj(p, obj_path, scale, mass, obj_t, obj_q):
    base_visualid = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        rgbaColor=None,
        meshScale=list(scale)
    )

    base_collisionid = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=list(scale),
    )

    new_obj = p.createMultiBody(mass,
                                base_collisionid,
                                base_visualid,
                                obj_t,
                                obj_q
                                )

    return new_obj

def load_object(p, path, pos, rot, scale, action_type):

    num_obj = len(glob.glob(path + "/model*.obj"))
    chosen_int = np.random.randint(0, num_obj)
    obj_path = path + "/model{0}.obj".format(chosen_int)

    if os.path.exists(path + "/texture{0}.png".format(chosen_int)):
        tex_path = path + "/texture{0}.png".format(chosen_int)
    else:
        tex_path = path + "/texture{0}.jpg".format(chosen_int)
    box_id = create_obj(p, obj_path, scale, 0.5, pos, Rot.from_rotvec(rot).as_quat())

    for i in range(480):
        p.stepSimulation()
    if action_type != "distractor":
        p.changeDynamics(box_id, -1, mass = 0)

    texture_id = p.loadTexture(tex_path)
    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
    return box_id

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return np.array(np.multiply(mask, 255)).astype(np.uint8)

def create_scene(p, path, pos, scale, camera_config, action_type, target_bbox):
    rot_q = Rot.from_rotvec([0,0,np.random.uniform(-3.14, 3.14)]).as_quat()

    rot_m = Rot.from_quat(rot_q).as_matrix()
    final_M = Rot.from_matrix(rot_m @ Rot.from_rotvec([1.57, 0, 0]).as_matrix()).as_rotvec()

    obj_id = load_object(p, path, pos, final_M, scale, action_type)

    if action_type=="distractor":
        min_x = p.getAABB(obj_id)[0][0]
        min_y = p.getAABB(obj_id)[0][1]
        max_x = p.getAABB(obj_id)[1][0]
        max_y = p.getAABB(obj_id)[1][1]
        current_bbox = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]

        for each_point in current_bbox:
            for each_target in target_bbox:
                discard = check_inside(each_point, each_target)
                if discard:
                    p.removeBody(obj_id)
                    return None, None, None, None
        p.changeDynamics(obj_id, -1, mass=0)

    # get rgbd images
    new_color, new_depth, segm = render_camera(p, camera_config)
    mask = np.zeros_like(segm)
    mask[segm==obj_id] = 255
    return new_color, new_depth, mask, obj_id


def get_obj_names():
    epic_names = ['corncob',  'scissor', 'pancake', 'candle',
                  'board', 'envelope',  'capers',  'squash', 'chopstick',
                   'mortar',  'paper',  'button', 'kettle',
                   'scale', 'corns',   'trousers',
                  'sweetcorn', 'pots', 'towel', 'cobs', 'machine', 'plastic', 'bottle',  'onion',
                  'cayenne', 'garlic', 'vegetables', 'skimmer', 'cling',  'peeler',
                  'cooking pan',  'cake', 'rings', 'noodles in a bowl',   'heater',
                   'knives',   'plastic dustpan',   'grill',
                  'refrigerator',  'rocket', 'cupboard', 'pancakes',
                   'rug', 'risotto',     'mushroom',
                     'a cup of smoothie', 'aubergine', 'a cup of coffee',
                  'broccoli',   'sock', 'fruit', 'nutella', 'tofu on a plate',
                   'brush',    'cucumber',   'glove',
                   'courgettes', 'shirt', 'cube', 'pepper on the plate', 'egg',
                  'dumplings', 'alarm', 'avocado',  'cap',   'coke can',
                   'toaster',  'crab', 'spill', 'bacon on a plate', 'scissor', 'pot', 'sushi',
                   'ham', 'pesto', 'tee', 'granola', 'sticker', 'peach',  'tortilla', 'teabag',
                  'spray',  'sieve', 'air', 'masher', 'teaspoon', 'jar',  'herb',
                     'sandwich on a plate',
                    'dish cloth',  'cheese on a plate',
                  'stove',  'teaspoon', 'strainer',
                   'waffle on a plate',  'ciabatta', 'spatula', 'knife', 'coffee mug', 'heart', 'sprout',
                  'bag', 'locker','shell', 'lime', 'phone',
                    'tray', 'dustbin',
                  'mat',  'turmeric', 'foil', 'yoghurt', 'spinach',
                  'onions',  'strainer', 'ceramic bowl',   'mozzarella', 'egg',
                  'spice', 'almond',   'rag', 'nesquik',
                   'tray',  'coffee cup', 'pepper on a plate',
                    'shell', 'flatware', 'poster', 'sandwiches', 'grape',
                 'basket', 'rind',  'leaves',  'courgette', 'bag',
                  'lettuce', 'cookies',  'lemon', 'silver spoon', 'stirrer', 'blueberries',
                  'watch',  'banana',  'trivet',
                   'nozzle', 'shelf', 'berries', 'bananas',  'oregano', 'cherry', 'chocolate',
                 'cube', 'plug',
                   'drainer',  'raisins',  'blender', 'basil',
                    'persil', 'cabbage on a plate',
                  'package',  'cork', 'wooden plate', 'grape on a plate',  'pear on a plate',
                   'popcorn',   'burger on a plate',  'sprout on a plate',
                  'spoon',  'leek', 'fish on a plate', 'cob on a plate',  'papers',
                     'eyeglasses', 'wooden cabinet', 'pizza on a plate', 'pie on a plate',
                  'glove', 'soap',   'eggshell', 'olives on a plate',
                  'microwave',    'lime',
                   'fraiche', 'mitt',
                  'sausage', 'bread',  'sponge', 'ladle', 'cupboards',
                  'cutlery',  'grinder', 'carrot veggie', 'maple',
                     'chicken statue',
                      'sponge',
                  'honey bottle',  'apple fruit', 'fridge', 'carafe',  'ginger',
                  'pineapple',  'tomato', 'leaf', 'sharpener',
                   'cilantro', 'coconut',  'grater', 'moon',  'wok',
                   'silverware','bulb',
                  'olive',  'fork',
                  'bucket', 'freezer','melon', 'mushroom',
                  'pitcher',  'dishwasher', 'peach',  'teapot',
                  'cookie', ]

    lvis_names = ['small air conditioner', 'airplane toy', 'alarm clock' , 'alligator', 'almond',
                  'ambulance toy',
                  'apple fruit',  'apricot fruit',
                   'armband', 'armchair',   'artichoke', 'realistic trash can',
                  'ashtray',
                  'asparagus',  'avocado',    'baboon statue',
                   'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet',
                  'basket ball',
                   'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage',
                   'barge', 'barrel', 'barrette',  'baseball_base', 'baseball', 'baseball_bat',
                  'baseball_cap', 'baseball_glove', 'basketball', 'bass_horn',
                  'bath_towel', 'bathrobe', 'battery', 'beachball',
                  'beanbag','bear','cow statue',
                  'beer bottle',
                  'beer can', 'beetle toy',  'red bell pepper on the plate', 'leather belt', 'belt buckle', 'bench', 'beret',  'Bible book',
                  'bicycle toy',  'binocular', 'bird statue', 'birdfeeder',  'birdcage',
                  'birdhouse', 'birthday_cake',  'pirate_flag', 'black_sheep statue', 'blackberry',
                   'blender', 'blimp', 'blinker','blueberry', 'gameboard', 'boat',
                  'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book',
                  'booklet', 'bookmark', 'boot', 'bottle', 'bottle_opener', 'bouquet',
                  'bow_(decorative_ribbons)', 'bow-tie',
                  'boxing_glove',  'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread',
                  'bridal_gown',  'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprout',
                  'bubble gum',
                  'bucket',  'bull statue', 'bulldog statue', 'bulldozer', 'bullet_train', 'bulletin_board',
                  'bulletproof_vest', 'bullhorn', 'burrito', 'bus',
                   'butterfly', 'button', 'yellow cab taxi', 'a wooden cabinet', 'locker', 'cake',
                  'calculator',  'camcorder', 'camel', 'camera', 'camera_lens', 'camper vehicle',
                  'can',
                  'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'canister',
                  'canoe',
                  'cantaloup', 'cap', 'bottle cap',
                   'car_battery',
                  'cargo_ship toy', 'carnation', 'horse carriage', 'carrot for food', 'tote bag', 'cart',
                  'casserole on a plate',  'cat toy', 'cauliflower', 'CD player',
                    'chaise longue', 'chalice', 'chandelier',
                  'checkbook',
                  'checkerboard', 'cherry', 'chessboard', 'chicken', 'chickpea',
                  'chinaware', 'poker_chip', 'chocolate_bar', 'chocolate_cake',
                  'chocolate_mousse', 'choker', 'chopstick', 'Christmas_tree',
                  'cigarette',  'cistern', 'clarinet', 'clasp',
                  'clementine', 'clip', 'clipboard',  'cloak',
                  'clock',
                  'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat hanger',
                  'coatrack',
                   'coconut',  'coffeepot',
                  'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
                  'compass', 'computer keyboard',  'cone',
                   'cookie',
                   'cowboy hat', 'crab',  'crayon',
                  'cream_pitcher', 'crouton', 'crow',
                  'crown',
                  'crucifix', 'cruise ship toy', 'police cruiser toy', 'crumb', 'crutch',  'cube', 'cucumber',
                  'cufflink', 'ceramic cup',  'cupboard', 'cupcake', 'hair_curler', 'curling_iron',
                   'cylinder', 'dagger', 'dalmatian statue',
                  'deer statue',
                    'diaper',   'dinghy',
                  'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser',
                  'Dixie_cup', 'dog statue',   'dollar', 'dollhouse', 'dolphin statue',
                  'doorknob',
                   'doughnut', 'dove', 'dragonfly', 'partially opened drawer', 'wooden drawer',  'dress', 'dress_hat',
                  'dresser',  'drone',
                  'duct_tape',  'dumbbell',  'eagle statue', 'earphone', 'earplug',
                  'earring',
                  'easel', 'eclair', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant',
                  'refrigerator', 'elephant statue', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon',
                  'faucet',
                  'fedora', 'ferret statue', 'Ferris_wheel', 'ferry', 'fig fruit', 'file_cabinet',
                   'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug',
                 'fish toy',  'fishing_rod', 'flag',

                  'flower_arrangement',  'foal', 'folding chair', 'American football',
                  'football_helmet',  'fork', 'forklift',   'freshener',
                  'frisbee',
                  'frog',  'frying_pan',   'futon',
                  'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle statue',
                  'generator',
                  'giant_panda statue', 'ginger', 'giraffe statue', 'cincture',  'globe',
                  'glove',
                  'goat statue', 'goggle', 'goldfish', 'golf_club', 'golfcart', 'goose toy', 'gorilla statue', 'gourd',
                   'grater', 'gravy_boat', 'green_onion', 'griddle', 'grill',
                    'guitar', 'gun', 'hairbrush', 'hairnet', 'hairpin',
                  'hamburger', 'hammer', 'hammock', 'hamper', 'hamster statue', 'hair_dryer',  'hand_towel',
                   'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat',
                  'hatbox',
                  'veil toy',  'headscarf', 'headset',
                  'heart',
                  'heater', 'helicopter toy', 'helmet', 'heron', 'hinge',  'hockey_stick',
                  'hog',
                  'home_plate_(baseball)',  'hook', 'hookah', 'hornet', 'horse statue', 'hose',
                  'hot-air_balloon', 'hotplate',  'hourglass', 'houseboat', 'hummingbird statue',
                  'polar_bear', 'icecream', 'popsicle',  'ice_skate', 'igniter', 'inhaler',
                  'iPod',
                   'jar', 'jean', 'jeep',
                  'jet_plane',  'joystick',  'kayak',   'kettle', 'key',
                  'keycard',   'kite',  'kiwi fruit',
                  'knee_pad',
                  'knife',  'knob',  'koala',  'ladder', 'ladle',
                  'ladybug',  'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern',
                  'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',  'Lego',
                  'legume',
                  'lemon', 'lemonade', 'lettuce', 'license_plate',  'life_jacket', 'lightbulb',
                  'lime', 'limousine', 'lion statue', 'lip_balm', 'lizard',
                  'speaker_(stero_equipment)', 'loveseat', 'machine_gun',  'magnet',
                   'mallard', 'mallet', 'mammoth', 'manatee',
                  'manhole',
                  'map', 'marker', 'martini', 'mascot',  'masher', 'mask',
                  'matchbox', 'measuring_cup',  'meatball', 'melon',
                  'microphone',
                  'microscope', 'microwave_oven',   'milk_can',  'minivan', 'mint_candy',
                  'mirror', 'mitten',  'money',
                  'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle',
                  'computer mouse', 'mousepad', 'muffin', 'ceramic mug', 'mushroom',
                  'musical_instrument',
                   'necklace', 'necktie',  'newspaper',
                  'nightshirt',  'notebook', 'notepad',
                    'oil_lamp',   'onion',
                  'orange',  'ostrich statue', 'owl statue',
                  'packet',
                  'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajama', 'palette',
                   'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel',
                   'parachute', 'parakeet',  'parasol',  'parka',
                  'parking_meter', 'parrot toy',  'passport', 'pastry',
                   'peach', 'peanut butter jar', 'pear on a plate',
                   'pelican statue', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum',
                  'penguin statue', 'pennant', 'penny_(coin)', 'pepper on a plate', 'pepper_mill', 'perfume bottle',
                  'piano toy', 'pickle', 'pickup_truck', 'pie on a plate',
                  'pigeon statue',
                  'plastic piggy bank', 'pillow',  'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel',
                  'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza on a plate',
                    'platter', 'playpen', 'plier', 'plow_(farm_equipment)', 'plume',
                  'pocket_watch',
                  'pocketknife',
                  'postcard', 'poster',  'flowerpot', 'potholder',
                  'pottery', 'pouch',   'pretzel', 'printer', 'projectile_(weapon)', 'projector',
                  'propeller', 'prune', 'pudding',  'puffin', 'pug dog statue', 'pumpkin',
                   'rabbit', 'race car toy', 'racket',
                  'radio_receiver', 'radish', 'raft',  'raincoat', 'ram_(animal)', 'raspberry', 'rat',
                    'record_player',
                  'remote_control', 'rhinocero',  'rifle', 'ring', 'river_boat', 'road_map', 'robe',
                  'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin',
                  'rubber_band',  'plastic shopping bag',
                   'saddlebag',    'salad_plate',
                   'sandwich', 'satchel',
                  'saxophone', 'scarecrow', 'scarf',
                  'school_bus', 'scissor', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture',
                  'seabird statue', 'seahorse toy', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo bottle', 'shark',
                  'sharpener',
                  'shaver', 'shawl', 'shears', 'sheep statue',
                  'shield', 'shirt', 'shoe', 'shopping bag', 'shopping_cart', 'short_pants',
                  'shoulder bag',
                   'shower_head', 'shredder_(for_paper)',  'silo',
                   'skateboard', 'skewer', 'ski', 'ski boot',  'skirt', 'skullcap',
                  'sled',
                'snake toy', 'snowboard', 'snowman',
                    'space_shuttle_toy',
                  'spatula',
                 'spice rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear',
                   'squirrel statue',  'stapler', 'starfish',
                  'statue_(sculpture)',
                   'steering_wheel',  'stereo sound_system',
                  'stirrer', 'stirrup',  'stop sign', 'brake_light', 'stove', 'strainer', 'strap',
                  'straw_(for_drinking)', 'strawberry',  'stylus',
                  'subwoofer', 'kitchen paper towel',
                  'sunflower', 'sunglasses', 'sunhat', 'barrow'
                  'surfboard',
                  'sushi', 'mop', 'sweatband', 'sweater',
                  'sword',
                  'syringe',  'table_lamp', 'tachometer',
                   'taillight', 'tambourine',
                  'tape measure', 'tassel', 'tea bag',
                  'teacup', 'teakettle', 'teapot', 'teddy bear', 'telephone', 'telephone booth',
                   'tennis ball', 'tennis racket', 'tequila',
                  'thermometer', 'thermos bottle', 'thermostat', 'thimble',  'thumbtack', 'tiara', 'tiger toy',
                    'tissue_paper',  'toaster',
                  'toaster_oven',   'tomato', 'toolbox', 'toothbrush', 'toothpaste',
                  'toothpick',  'tow truck toy',  'towel rack', 'toy',
                  'tractor_(farm_equipment)',
                  'traffic_light', 'dirt_bike', 'trailer_truck', 'train', 'trampoline', 'tray',
                  'trench_coat',  'tricycle', 'tripod', 'trousers', 'truck',
                  'truffle chocolate',  'turban',  'turnip', 'turtle',
                   'typewriter', 'umbrella', 'unicycle',
                  'vacuum cleaner', 'vase', 'videotape',  'violin',
                  'volleyball',  'waffle', 'waffle_iron',
                  'wall_clock',
                  'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'watch',
                  'water bottle',
                  'water cooler', 'water faucet', 'water heater' 'water gun', 'water scooter',
                  'water tower',  'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring',
                 'wheel', 'wheelchair', 'whistle', 'wig', 'wind_chime', 'windmill',
                  'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass',
                   'wok',  'wooden spoon', 'wreath', 'wrench', 'wristband', 'wristlet',
                  'yacht toy', 'yogurt',  'zebra statue', 'zucchini']
    final_names = epic_names + lvis_names
    set(final_names)
    return final_names