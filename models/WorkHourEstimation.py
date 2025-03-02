# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:18:43 2024

@author: Murad
"""

import os
import cv2
import av
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from models.FaceDetector import FaceDetector
from models.FaceRecognition import FaceRecognition
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

class PersonDataStream:
    def __init__(self, args):
        self.args = args
        self.Facedetector = FaceDetector(self.args)
        self.Facerecognizer = FaceRecognition(self.args)
        self.Facerecognizer.__load_checkpoint__(checkpoint_dir = './models/face_recognition_model.yml')
        self.skip_frames = self.args.wh_skip_frames # skip_frames n means skipping n frames
        self.root_data = self.args.root_data
        self.dir_timestamp = './outputs/Work_Hour_Estimation/'
        self.intruder_min_score = self.args.intruder_min_score
        self.hourly_presence_threshold = self.args.hourly_presence_threshold
        os.makedirs(self.dir_timestamp, exist_ok = True)
        
    def run(self):
        video_spec = pd.read_csv(os.path.join(self.root_data, 'video_information', 'video_spec.csv'))
        person_videos_spec = video_spec[video_spec['face'] == 1]
        timestamp_dic = {'timestamp': [], 'video_name': [], 'label': [], 'score': [], 'frame': [], 'person_num': []}
        
        for i, person_video_spec in person_videos_spec.iterrows():
            video_path = os.path.join(self.root_data, 'videos', person_video_spec['video_name'])
            video_start_time = person_video_spec['start_time']
            video_end_time = person_video_spec['end_time']
            video_fps = person_video_spec['fps']
            time_delta = pd.Timedelta(seconds = 1/video_fps)
            
            container = av.open(video_path)
            current_time = pd.Timestamp(video_start_time)
            
            for i, frame in enumerate(container.decode(video = 0)):
                if i % (self.skip_frames + 1) != 0:
                    current_time = current_time + time_delta
                    continue
                
                image = frame.to_image()
                image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                _, faces = self.Facedetector.Detector.detect(image_array)
                
                if faces is not None: 
                    for n,  face in enumerate(faces):
                        box = face[:4].astype(np.int32)
                        x, y, w, h, score = box[0], box[1], box[2], box[3], face[-1]
                        cropped_face = image_array[y: y + h, x: x + w]
                        
                        label, score = self.Facerecognizer.recognizePerson(cropped_face)
                        timestamp_dic['timestamp'].append(current_time)
                        timestamp_dic['video_name'].append(person_video_spec['video_name'])
                        timestamp_dic['label'].append(label)
                        timestamp_dic['score'].append(score)
                        timestamp_dic['frame'].append(i)
                        timestamp_dic['person_num'].append(n)
                current_time = current_time + time_delta    
                
        timestamp_dic = pd.DataFrame(timestamp_dic)
        timestamp_dic.to_csv(os.path.join(self.dir_timestamp, 'timestamp_{}_{}.csv'.format(self.args.score_threshold, 
                                                                                           self.skip_frames)), index = False)
        print('timestamp is saved in: {}'.format(self.dir_timestamp))
        return timestamp_dic
    
    
    def work_hour_estimate(self, run = True):
        if run:
            timestampinfo = self.run()
        else:
            timestampinfo = pd.read_csv(os.path.join(self.dir_timestamp, 
                                                     'timestamp_{}_{}.csv'.format(self.args.score_threshold, self.skip_frames)))
        # intruder removing
        data = timestampinfo[timestampinfo['score']< self.intruder_min_score]
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.floor('H')
        hourly_presence_table = data.pivot_table(index = 'hour',
                                                 columns = 'label',
                                                 values = 'video_name', 
                                                 aggfunc = 'count', 
                                                 fill_value = 0 
                                                 )
        
        hourly_presence_matrix = (hourly_presence_table >= self.hourly_presence_threshold).astype(int)
        attn_name = 'hourly_person_attendance_{}_{}_h{}.csv'.format(self.args.score_threshold, 
                                                                    self.skip_frames, 
                                                                    self.hourly_presence_threshold)
        hourly_presence_matrix.to_csv(os.path.join(self.dir_timestamp, attn_name))
        
        # ########## PLOTTING THE HOURLY PRESENCE #############################
        hourly_presence_matrix.index = pd.to_datetime(hourly_presence_matrix.index)
        single_day_hourly_presence = hourly_presence_matrix.loc['2024-08-05': '2024-08-06']
        range1 = single_day_hourly_presence.index.min() # starting date for plot
        range2 = single_day_hourly_presence.index.min() + pd.Timedelta(hours = 48) # ending date for plot
        full_range = pd.date_range(start = range1, end = range2, freq = '1H') # '1T'
        single_day_hourly_presence = single_day_hourly_presence.reindex(full_range)
        single_day_hourly_presence = single_day_hourly_presence.fillna(0)
        
        df = single_day_hourly_presence 
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d %H:%M')
        df = single_day_hourly_presence
        df_transposed = df.T
        df_transposed = df_transposed.loc[[1, 12]] # selecting person
        
        plt.figure(figsize=(15, 6))
        binary_cmap = ListedColormap(["#FFF8DC", "#4682B4"]) # ListedColormap(["lightyellow", "green"])
        sns.heatmap(df_transposed, cmap = binary_cmap, cbar = False) #cbar_kws={'label': 'Presence (1=Present, 0=Absent)'})
        legend_labels = [mpatches.Patch(color = "#FFF8DC", label = "0 (Absent)"), 
                         mpatches.Patch(color = "#4682B4", label = "1 (Present)")
                         ]
        plt.legend(handles = legend_labels, title = "Legend", bbox_to_anchor = (0.9, 1), loc = 'upper left')

        plt.title("Hourly Presence Timeline")
        plt.ylabel("Crew ID")
        plt.xlabel("Time (Date and Hour)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        # plt.close('all')
        return
        
    
    