# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:09:55 2024
@author: Murad
SISLab, USF
mmurad@usf.edu
"""
import os
import cv2
import av
import numpy as np


class FaceDetector:
    def __init__(self, args):
        self.args = args
        self.data_path = os.path.join(self.args.root_data, self.args.video)
        self.skip_frames = self.args.fd_skip_frames
        self.output_path = './outputs/'
        self.Detector = cv2.FaceDetectorYN.create(self.args.face_detection_model,
                                                  "",
                                                  (self.args.width, self.args.height),
                                                  self.args.score_threshold,
                                                  self.args.nms_threshold,
                                                  self.args.top_k
                                                  )
        os.makedirs(self.output_path, exist_ok = True)
        
        
    def __get_all_files_dir__(self):
        data_dir = self.data_path
        file_paths = []
        for sp in os.listdir(data_dir):
            sub_folder_path = data_dir + sp + '/'
            for lp in os.listdir(sub_folder_path):
                file_path = sub_folder_path + lp
                file_paths.append(file_path)
        return file_paths


    def detect_face(self): 
        """
        skip_frames 0 means no skipping
        skip_frames n means skipping n frames
        """
        skip_frames = self.skip_frames
        file_paths = self.__get_all_files_dir__()
        start_idx = 0
        end_idx = -1
        os.makedirs(out_dir := os.path.join(self.output_path, 'faces'), exist_ok = True)
        
        for k, file_path in enumerate(file_paths[start_idx: end_idx], start = start_idx):
            container = av.open(file_path)
            print('Processing: {}'.format(file_path))
            try:
                for i, frame in enumerate(container.decode(video = 0)):
                    # skipping frames
                    if i % (skip_frames + 1) != 0:
                        continue
                    
                    image = frame.to_image()
                    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    _, faces = self.Detector.detect(image_array)
                    if faces is not None: # len(faces)
                        print('face-found')
                        for n,  face in enumerate(faces):
                            box = face[:4].astype(np.int32)
                            x, y, w, h, score = box[0], box[1], box[2], box[3], face[-1]
                            cropped_face = image_array[y: y + h, x: x + w]
                            image_name = file_path.split('.')[1].replace('/', '_') + '_' + str(i) + '_' + str(n) + '.jpg'
                            cv2.imwrite(out_dir + image_name, cropped_face)
            except Exception:
                with open(os.path.join(out_dir, 'error.txt'), "a") as f:
                    info = file_path + '_' + 'frame_' + str(i)
                    f.write(info + "\n")
                continue
        print('Deteced faces have been saved in {}'.format(out_dir))
            
            
    def save_full_frame(self):
        skip_frames = self.skip_frames
        file_paths = self.__get_all_files_dir__()
        
        for file_path in file_paths:
            container = av.open(file_path)
            video_name = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path)).split('.')[0]
            s = 0
            os.makedirs(out_dir := self.output_path + 'frames/' + video_name + '/' + str(s) + '/', exist_ok = True)
            try:
                for i, frame in enumerate(container.decode(video = s)):
                    if i % (skip_frames + 1) != 0:
                        continue
                    image = frame.to_image()
                    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_name = file_path.split('.')[1].replace('/', '_') + '_' + str(i) + '.jpg'
                    cv2.imwrite(out_dir + image_name, image_array)
            except: 
                print('error: {}'.format(out_dir))
                os.rmdir(out_dir)
        print('Saved frame images directory: {}'.format(self.output_path + 'frames/'))
        return
    
    
    def __check_frame_size__(self):
        file_paths = self.__get_all_files_dir__()
        start_idx = 0 
        error_idx, error_files = [], []
        anom_idx, anom_files = [], []
        for k, file_path in enumerate(file_paths[start_idx:], start = start_idx):
            container = av.open(file_path)
            try:
                for i, frame in enumerate(container.decode(video = 0)):
                    image = frame.to_image()
                    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    print(k, file_path)
                    if (image_array.shape[0] != 288) or (image_array.shape[1] != 352):
                        print('found video-not in desired shape')
                        anom_idx.append(k)
                        anom_files.append(file_path)
                    break
            except:
                print('{}, Error'.format(k))
                error_idx.append(k)
                error_files.append(file_path)
        return error_idx, error_files, anom_idx, anom_files