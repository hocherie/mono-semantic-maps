import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from shapely.strtree import STRtree
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from matplotlib import pyplot as plt 
import cv2

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, transform, \
    encode_binary_labels
import src.data.nuscenes.utils as nusc_utils


def process_scene(nuscenes, map_data, scene, config):

    # Get the map corresponding to the current sample data
    log = nuscenes.get('log', scene['log_token'])
    scene_map_data = map_data[log['location']]

    # Iterate over samples
    first_sample_token = scene['first_sample_token']
    for sample in nusc_utils.iterate_samples(nuscenes, first_sample_token):
        process_sample(nuscenes, scene_map_data, sample, config)


def process_sample(nuscenes, map_data, sample, config):

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)

    # Iterate over sample data
    for camera in nusc_utils.CAMERA_NAMES:
        sample_data = nuscenes.get('sample_data', sample['data'][camera])
        process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, config)


def process_sample_data(nuscenes, map_data, sample_data, lidar, config):

    # Only process for cam front
    cam_img_filename = sample_data['filename']
    cam_folder = cam_img_filename.split('/')[1]
    if cam_folder == "CAM_FRONT":


        # Render static road geometry masks
        map_masks = nusc_utils.get_map_masks(nuscenes, 
                                            map_data, 
                                            sample_data, 
                                            config.map_extents, 
                                            config.map_resolution)
        
        # Render dynamic object masks
        obj_masks = nusc_utils.get_object_masks(nuscenes, 
                                                sample_data, 
                                                config.map_extents, 
                                                config.map_resolution)
        # map_masks: (6,196, 200)    
        # obj_masks: (11, 196,200)                                    
        masks = np.concatenate([map_masks, obj_masks], axis=0) # (15, 196, 200)


        plt_row = 3
        plt_col = 6

        masks_title = nusc_utils.STATIC_CLASSES + nusc_utils.NUSCENES_CLASS_NAMES + ['others']

        # import pdb; pdb.set_trace()
        # print(sample_data['filename'])
        # import pdb; pdb.set_trace()
        cam_img_filepath = config.dataroot + '/' + sample_data['filename']
        cam_img = cv2.imread(cam_img_filepath)
        cv2.imwrite(os.path.join(os.path.expandvars(config.label_root), 'road_' + str(sample_data['timestamp']) + '_img.png'),
                    cam_img)
        # import time 
        # start_time = time.time()
        # plt.clf()
        # for map_i in range(masks.shape[0]):
        #     plt.subplot(plt_row, plt_col, map_i+1)

        #     plt.imshow(masks[map_i].astype(np.int32)*255)
        #     plt.title(masks_title[map_i])
        # plt.savefig(os.path.join(os.path.expandvars(config.label_root), 'all_' + 
        #                         sample_data['token'] + '.png'))
        cv2.imwrite(os.path.join(os.path.expandvars(config.label_root), 'road_' + str(sample_data['timestamp']) + '.png'),
                        (masks[4] | masks[5]).astype(np.int32)*255)
        # print(sample_data['timestamp'])
        # print("Time taken (s):", np.round(time.time() - start_time, 2))

        # import pdb; pdb.set_trace()
        # Ignore regions of the BEV which are outside the image
        # sensor = nuscenes.get('calibrated_sensor', 
        #                       sample_data['calibrated_sensor_token'])
        # intrinsics = np.array(sensor['camera_intrinsic'])
        # masks[-1] |= ~get_visible_mask(intrinsics, sample_data['width'], 
        #                                config.map_extents, config.map_resolution)
        
        # # Transform lidar points into camera coordinates
        # cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
        # cam_points = transform(np.linalg.inv(cam_transform), lidar)
        # masks[-1] |= get_occlusion_mask(cam_points, config.map_extents,
        #                                 config.map_resolution)
        
        # # Encode masks as integer bitmask
        # labels = encode_binary_labels(masks)

        # # Save outputs to disk
        # output_path = os.path.join(os.path.expandvars(config.label_root),
        #                         sample_data['token'] + '.png')
        # Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)


def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in nusc_utils.STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)

        
        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)
    
    return map_data





if __name__ == '__main__':

    # Load the default configuration
    config = get_default_configuration()
    config.merge_from_file('configs/datasets/nuscenes.yml')

    # Load NuScenes dataset
    dataroot = os.path.expandvars(config.dataroot)
    nuscenes = NuScenes(config.nuscenes_version, dataroot)

    # Preload NuScenes map data
    map_data = { location : load_map_data(dataroot, location) 
                 for location in nusc_utils.LOCATIONS }
    
    # Create a directory for the generated labels
    output_root = os.path.expandvars(config.label_root)
    os.makedirs(output_root, exist_ok=True)
    
   # print(nuscenes.scene)
    # Iterate over NuScene scenes
    print("\nGenerating labels...")
    for scene in tqdm(nuscenes.scene):
        process_scene(nuscenes, map_data, scene, config)




    


    





    

