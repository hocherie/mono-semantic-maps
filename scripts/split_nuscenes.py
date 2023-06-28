"""
Split intermeditate-processed NuScenes folder 
to train / test split given a list of test location points.
"""
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image


# dev process: make functions here, then debug main function with python notebook 

def say_hi():
    print('hi tigu')


def get_scene_tokens(nusc, location, remove_blacklist=True):
    """
    Given a location, return a list of scene tokens in that location.
    """
    # Get list of logs in Nuscenes data object in a given map
    log_tokens = [log['token'] for log in nusc.log if log['location'] == location]

    # Filter scenes given logs.
    scene_tokens_in_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
    scene_blacklist = [499, 515, 517] # 2 are in boston-seaport

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



if __name__ ==' __main__':
    pass 

    # # initialize variables
    # input_folder_path = '/home/kevin/Desktop/intermediate-processed-nuscenes' 
    # # TODO: replace with yaml file and parser 

    # # initialize NuScenes dataset (full dataset usually takes 1 min to load)
    # dataroot = '/ocean/projects/cis220039p/cherieho/data/datasets/nuscenes_full'
    # nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)


    # for each location in yml file (tqdm: processing ...), 
    # [x]fx(): get list of scene_tokens in location 
    
    # for each scene in location given by scene_tokens
    ### [x]fx(): make poselist_dict dictionary for location {key: scene_token, value: poses_in_scene}

    ### [x] fx(): return scene tokens that are within dist_thresh of test_location_point 
    ### [ ] plot trajectories of scene in location, colored based on train / test 