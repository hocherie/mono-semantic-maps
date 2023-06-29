"""
Split intermeditate-processed NuScenes folder 
to train / test split given a list of test location points.
"""
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image
import yaml
import shutil

import sys 
import os 


# dev process: make functions here, then debug main function with python notebook 


scene_blacklist = [499, 515, 517] # 2 are in boston-seaport

def get_scene_tokens(nusc, location, remove_blacklist=True):
    """
    Given a location, return a list of scene tokens in that location.
    """
    # Get list of logs in Nuscenes data object in a given map
    log_tokens = [log['token'] for log in nusc.log if log['location'] == location]

    # Filter scenes given logs.
    scene_tokens_in_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
    

    for scene_token in scene_tokens_in_location:
        scene_name = nusc.get('scene', scene_token)['name']
        scene_id = int(scene_name.replace('scene-', ''))

        if scene_id in scene_blacklist:
            print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)
            if remove_blacklist:
                print('Removing %s from list of scenes.' % scene_name)
                scene_tokens_in_location.remove(scene_token)
            else:
                print('remove_blacklist is set to False, not removing %s from list of scenes.' % scene_name)

    # TODO: add a toggle to remove scenes in blacklist
    return scene_tokens_in_location

def get_pose_dict_given_scenes(nusc, scene_tokens_in_location):
    """
    Given a list of scene tokens, return a dictionary of scene tokens and the poses.

    dictionary: {key: scene_token, value: poses_in_scene [np.array of shape (40-ish,3)]}
    """
    poselist_dict = {}

    for scene_token in tqdm(scene_tokens_in_location, desc='Get pose from location'):
        # For each sample in the scene, store the ego pose.
        pose_list = []
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
        for sample_token in sample_tokens:
            sample_record = nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            pose_list.append(pose_record['translation'])
        # Add to dictionary
        poselist_dict[scene_token] = np.array(pose_list)        

    return poselist_dict

def get_train_test_split_dict_from_testpoint(poselist_dict, test_point, dist_threshold):
    """
    Get a split dictionary with key as scene tokens, and value as 0/1 whether or not in test set.
    """
    split_dict = {}
    num_train = 0
    num_test = 0
    print('getting test split')

    for scene_token, pose_list in tqdm(poselist_dict.items(), desc='Get train/test split'):
        # Get closest distance over the scene to test point
        min_dist = np.min(np.linalg.norm(pose_list[:, :2] - test_point[:2], axis=1))
        # Assign to train or test set.
        if min_dist > dist_threshold: # add to train, if far enough from test point
            split_dict[scene_token] = 0
            num_train += 1
        else:
            split_dict[scene_token] = 1
            num_test += 1

    print('# Total Scenes:', num_train + num_test)
    print('# Train Scenes: {num_train}, {perc_train}% of total'.format(num_train=num_train, perc_train=np.round(num_train/(num_train+num_test)*100)))
    print('# Test Scenes: {num_test}, {perc_test}% of total'.format(num_test=num_test, perc_test=np.round(num_test/(num_train+num_test)*100)))

    return split_dict
    
def plot_traj_colored_by_traintest(poselist_dict, split_dict, test_point, dist_threshold, location):
    """
    Plot trajectories of scenes in location, colored based on train / test.
    """
    # Plot trajectory, with each scene colored by whether or not in test
    for scene_token, pose_list in tqdm(poselist_dict.items(), desc='Plot trajectories'):
        if split_dict[scene_token] == 0:
            plt.scatter(pose_list[:, 0], pose_list[:, 1], c='b', s=0.2)
        else:
            plt.scatter(pose_list[:, 0], pose_list[:, 1], c='r', s=3)
    
    # Plot test point and threshold
    plt.plot(test_point[0], test_point[1], 'kx')
    circle = plt.Circle((test_point[0], test_point[1]), dist_threshold, color='k', fill=False)
    plt.gca().add_artist(circle)

    # Add legend
    plt.plot([], [], 'b', label='Train')
    plt.plot([], [], 'r', label='Test')
    plt.legend()

    # Add title given location
    plt.title('Trajectories in {map_location} within {dist_threshold}m of test point.'.format(map_location=location, dist_threshold=dist_threshold))

    # Add axis
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.show()

def get_location_and_traintest_given_sample_token(sample_token, nusc, split_dict_all_locations):
    """ Get location and whether it is in train or test given sample token.
    """

    # Get location from sample token
    scene_token = nusc.get('sample', sample_token)['scene_token']
    log_token = nusc.get('scene', scene_token)['log_token']
    location = nusc.get('log', log_token)['location']

    # Get if sample token should be in train or split folder
    # check if in split_dict_all_locations
    if location not in split_dict_all_locations.keys():
        print('Location %s not in split_dict_all_locations' % location)
        # is_test = None
    else:
        if scene_token not in split_dict_all_locations[location].keys():
            scene_name = nusc.get('scene', scene_token)['name']
            scene_id =  int(scene_name.replace('scene-', ''))
            print('Scene token %s with name %s not in split_dict_all_locations' % (scene_token, nusc.get('scene', scene_token)['name']))

            if scene_id in scene_blacklist:
                print('Scene is in scene_blacklist and likely should not have split information and should be skipped, returning is_test as None')
                is_test = None
            
        else:
            is_test = split_dict_all_locations[location][scene_token] 
    
    return location, is_test


def get_sample_token_list_from_folder(folder_path):
    """ Get list of sample tokens from folder path.

    Sample tokens are found by removing the .png extension from the file name.
    """
    cam_folder_path = folder_path + '/cam'
    map_folder_path = folder_path + '/map'
    sample_token_list_cam = []
    sample_token_list_map = []
    for filename in os.listdir(cam_folder_path):
        sample_token_list_cam.append(filename[:-4])
    for filename in os.listdir(map_folder_path):
        sample_token_list_map.append(filename[:-4])

    # assert that both list are the same
    assert sample_token_list_cam == sample_token_list_map, "sample_token_list_cam and sample_token_list_map are not the same"

    return sample_token_list_cam
    
def get_new_path_given_sample_token(sample_token, cam_or_map, nusc, split_dict_all_locations, output_folder_path):
    """ Get new path given sample token.

    New path is given by location and train/test split.
    """
    # Get location and train/test split
    location, is_test = get_location_and_traintest_given_sample_token(sample_token, nusc, split_dict_all_locations)

    # Get new path
    if is_test is None: # don't have is_test information, likely because skipped due to blacklist
        new_path = None 

    location_folder_path = output_folder_path + '/'+ cam_or_map + '/' + location + '/'
    if is_test == 0:
        new_path = location_folder_path + '/train/' + sample_token + '.png'
    else:
        new_path = location_folder_path + '/test/' + sample_token + '.png'

    return new_path

def get_orig_path_given_sample_token(sample_token, cam_or_map, input_folder_path):
    """ Get original path given sample token.

    Original path is given by input folder path and cam or map.
    """
    orig_path = input_folder_path + '/' + cam_or_map + '/' + sample_token + '.png'

    return orig_path

def get_paths_given_sample_token(sample_token, cam_or_map, nusc, split_dict_all_locations, input_folder_path, output_folder_path):
    """ Get original and new path given sample token.

    Original path is given by input folder path and cam/map.
    New path is given by location and train/test split and cam/map.
    """
    orig_path = get_orig_path_given_sample_token(sample_token, cam_or_map, input_folder_path)
    new_path = get_new_path_given_sample_token(sample_token, cam_or_map, nusc, split_dict_all_locations, output_folder_path)

    return orig_path, new_path

def create_output_folders(output_folder_path, split_config_dict):
    """Make output directories"""

    # Create output directories for each location 
    for location in split_config_dict.keys():
        for cam_or_map in ['cam', 'map']:
            for train_or_test in ['train', 'test']:
                children_folder_path = output_folder_path + '/'+ cam_or_map + '/' + location + '/' + train_or_test
                print(children_folder_path)
                os.makedirs(children_folder_path, exist_ok=True)

    print('Output directories created or already exists')

# if __name__ ==' __main__':
# generally trainval takes 1 minute to initialize
dataroot = '/ocean/projects/cis220039p/cherieho/data/datasets/nuscenes/nuscenes_full'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

# variables

# Test point decision rationale: 
# R1: target around 10% of total scenes being test
# R2: decent amount of corners and turns (not just straightaways)
split_config = '../configs/nuscenes_split/test_0628.yml'

with open(split_config, 'r') as file:
    split_config_dict = yaml.safe_load(file)

split_dict_all_locations = {}
for location in split_config_dict.keys():
    test_point = split_config_dict[location]['test_point']
    dist_threshold = split_config_dict[location]['dist_threshold']
    scene_tokens_in_location = get_scene_tokens(nusc, location)
    posedict = get_pose_dict_given_scenes(nusc, scene_tokens_in_location)
    split_dict = get_train_test_split_dict_from_testpoint(posedict, test_point, dist_threshold)
    split_dict_all_locations[location] = split_dict
    # plot_traj_colored_by_traintest(posedict, split_dict, test_point, dist_threshold, location)


input_folder_path = '/ocean/projects/cis220039p/cherieho/data/datasets/nuscenes/nuscenes_full_intermprocessed_alllog_camfront_map_0629'
output_folder_path = '/ocean/projects/cis220039p/cherieho/data/datasets/nuscenes/nuscenes_processed_camfrontmap_0629'

# Create output folder
create_output_folders(output_folder_path, split_config_dict)

# Given files in input folder, copy into new folder
sample_tokens = get_sample_token_list_from_folder(input_folder_path)
for sample_token in tqdm(sample_tokens):
    for cam_or_map in ['cam', 'map']:
        old_path_sample, new_path_sample = get_paths_given_sample_token(sample_token, cam_or_map, nusc, split_dict_all_locations, input_folder_path, output_folder_path)
        if new_path_sample is not None: # if it is valid. normally invalid, because is_test info does not exist
            # copy image from old path to new path
            shutil.copy(old_path_sample, new_path_sample)