# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:42:11 2024

@author: Murad
"""

import os
import av
from datetime import datetime, timedelta
import pandas as pd

def __get_all_files_dir__(data_dir):
    file_paths = []
    for sp in os.listdir(data_dir):
        sub_folder_path = data_dir + sp + '/'
        for lp in os.listdir(sub_folder_path):
            file_path = sub_folder_path + lp
            file_paths.append(file_path)
    return file_paths

def __get_mod_time__(video_dir):
    file_stats = os.stat(video_dir)
    mod_time = datetime.fromtimestamp(file_stats.st_mtime)
    return mod_time
    
def __get_fps__(video_dir):
    container = av.open(video_dir)
    fps = float(container.streams.video[0].average_rate)
    
    for n, _ in enumerate(container.decode(video = 0)):
        pass
    number_frames = n + 1
    duration_sec = number_frames / fps
    return fps, duration_sec

def __get_listdir_of_videos_which_has_person__():
    all_face_path_dir = 'D:/Research/MarineFaceDetection/1_FaceDetection_11_30_2024/Outputs_collection/Detected_face_12_2_2024'
    all_face_path = os.listdir(all_face_path_dir)
    final_paths = []
    for single_path in all_face_path:
        '''don't try to understand. just ignore'''
        single_path = single_path.replace('_file', '-file')
        single_path = single_path.replace('_data_', '')
        single_path = single_path.replace('.jpg', '')
        pathparts = single_path.split('-')
        part1 = pathparts[0]
        part2 = pathparts[1].split('_')[0]
        final_path = part1 + '/' + part2 + '.dat' 
        final_paths.append(final_path)
    unique_face_path = list(set(final_paths))
    return unique_face_path

def create_video_spec_csv(data_dir):
    videos_dir = __get_all_files_dir__(data_dir)
    L = len(videos_dir)
    video_spec = {'video_name': [], 'start_time': [], 'end_time': [], 'duration': [], 'fps': [], 'face': []}
    error = []
    
    person_dir = __get_listdir_of_videos_which_has_person__()
    for i, video_dir in enumerate(videos_dir):
        try:
            mod_time = __get_mod_time__(video_dir)
            end_time = mod_time + timedelta(hours = 15)
            fps, duration_sec = __get_fps__(video_dir)
            start_time = end_time - timedelta(seconds = duration_sec)
            
            end_time = pd.Timestamp(end_time)
            start_time = pd.Timestamp(start_time)
            
            vdir = video_dir.split('/')[-2] + '/' + video_dir.split('/')[-1]
            video_spec['video_name'].append(vdir)
            video_spec['start_time'].append(start_time)
            video_spec['end_time'].append(end_time)
            video_spec['duration'].append(duration_sec)
            video_spec['fps'].append(fps)
            if vdir in person_dir:
                video_spec['face'].append(1) # 1 means: it has face
            else:
                video_spec['face'].append(0) # It does not have face
        except:
            error.append(video_dir)
        
        if i % 500 == 0:
            print('Completed: {}'.format(i/L))
    return pd.DataFrame(video_spec), error
    
    
if __name__ == '__main__':
    data_dir = './data/videos/'
    video_spec, error_files = create_video_spec_csv(data_dir)
    
    
    