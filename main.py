# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:18:43 2024
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import argparse
from models.FaceDetector import FaceDetector
from models.FaceRecognition import FaceRecognition
from models.WorkHourEstimation import PersonDataStream
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, default = 'work_hour_estimation', choices = ['crew_recognition', 'work_hour_estimation', 'face_detection'])
    parser.add_argument('--root_data', type = str, default = './data/')
    parser.add_argument('--video', type = str, default = 'videos', help = 'Video data folder inside the root_data directory')
    parser.add_argument('--face_data', type = str, default = 'face_photos', help = 'Face images folder inside the data root directory')
    
    # Detector model params
    parser.add_argument('--face_detection_model', type = str, default = './models/face_detection_yunet_2023mar.onnx')
    parser.add_argument('--score_threshold', type = float, default = 0.8, help = 'Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type = float, default = 0.3, help =' Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type = int, default = 5000, help = 'Keep top_k bounding boxes before NMS.')
    parser.add_argument('--height', type = int, default = 288, help = 'Input image height')
    parser.add_argument('--width', type = int, default = 352, help = 'Input image width')
    parser.add_argument('--fd_skip_frames', type = int, default = 30, help = 'num of skipping frames')

    # Recognizer model params
    parser.add_argument('--min_face_images', type = int, default = 40, help = 'Minimum number of face images for face recognition')
    parser.add_argument('--seed', type = int, default = 42)
    
    # Work Hour Estimation Params
    parser.add_argument('--wh_skip_frames', type = int, default = 0, help = 'num of skipping frames')
    parser.add_argument('--intruder_min_score', type = float, default = 102.267, help = 'Minimum score for intruders')
    parser.add_argument('--hourly_presence_threshold', type = int, default = 1, help = 'Hourly number of presence should be greater than this')
    args = parser.parse_args()
    
    if args.task == 'face_detection':
        # ############### FACE DETECTOR ############################
        FD = FaceDetector(args)
        FD.detect_face() # DETECTING AND CROPPING THE FACE IMAGES FROM ALL THE VIDEOS
        # FD.save_full_frame() # SAVING THE FRAME IMAGES FROM ALL THE VIDEOS
        
    elif args.task == 'crew_recognition':
        # ############## FACE RECOGNITION MODEL TRAINING ##########
        FaceRecognizer = FaceRecognition(args)
        FaceRecognizer.train()
        FaceRecognizer.test_intruder_detection()
        
    elif args.task == 'work_hour_estimation':
        # ############## WORK HOUR ESTIMATION #####################
        PDS = PersonDataStream(args)
        PDS.work_hour_estimate()
    

    