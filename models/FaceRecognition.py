# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:13:39 2024
@author: Murad
SISLab, USF
mmurad@usf.edu
"""
import cv2
import os
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from utils.count_faces import countfaces
from utils.misc_functions import split_data, result_for_intruders, result

class FaceRecognition:
    def __init__(self, args):
        self.args = args
        self.min_face_images = self.args.min_face_images
        self.face_data = os.path.join(self.args.root_data, self.args.face_data)
        self.result_path = './outputs/Face_recognition_results/min_face_images_{}'.format(self.min_face_images)
        self.seed = self.args.seed
        self.model = cv2.face.LBPHFaceRecognizer_create(neighbors = 8)
        
        self.split_path = os.path.join(self.face_data, 'split_data')
        self.unsplit_path = os.path.join(self.face_data, 'unsplit_data')
        self.person_info = countfaces(self.unsplit_path)
        self.selected_person_list = self.person_info[self.person_info['number'] > self.min_face_images]['person'].values 
        self.intruder_list = self.person_info[self.person_info['number'] <= self.min_face_images]['person'].values
        print('Person: {}'.format(self.selected_person_list))
        print('Person number: {}'.format(self.person_info))
        os.makedirs(self.result_path, exist_ok = True)
        
        
    def __load_checkpoint__(self, checkpoint_dir = []):
        self.model.read(checkpoint_dir)
        
        
    def __preprocess_image__(self, img):
        if len(img.shape) > 2:  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        img = cv2.resize(img, (100, 100))
        img = cv2.equalizeHist(img)
        return img


    def __get_images_and_ids__(self, directory, shuffle = True):
        images = []
        ids = []
        
        for path, subdirnames, files in os.walk(directory):
            for f in files:
                id2 = os.path.basename(path)
                img = os.path.join(path, f)
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print("Failed to load image")
                    continue
                
                image = self.__preprocess_image__(image)
                images.append(image)
                ids.append(int(id2))
        
        if shuffle == True:
            combined = list(zip(images, ids))
            random.shuffle(combined)
            images, ids = zip(*combined)
            images = list(images)
            ids = list(ids)
        return images, ids


    def train(self):
        split_data(self.unsplit_path, self.split_path, self.selected_person_list, self.intruder_list, test_ratio = 0.2, val_ratio = 0.2, seed = 42, delete_dir = True) 
        os.makedirs(train_dir := os.path.join(self.split_path, 'train'), exist_ok = True)
        images, ids = self.__get_images_and_ids__(train_dir, shuffle = True)
        
        self.model.train(images, np.array(ids))
        self.model.save('./models/face_recognition_model.yml')
        return self.model


    def test(self, test_dir):
        self.__load_checkpoint__(checkpoint_dir = './models/face_recognition_model.yml')
        test_images, ids_true = self.__get_images_and_ids__(test_dir, shuffle = True)
        
        y_true = []
        y_pred = []
        score = []
        dic = {'id_true': [], 'id_pred': [], 'score': []}
        
        for (img, id_) in zip(test_images, ids_true):
            label, confidence = self.model.predict(img)
            y_true.append(id_)
            y_pred.append(label)
            score.append(confidence)
            print(f"Ground Truth: {id_}, Predicted: {label}, Confidence: {confidence}")
    
        dic['id_true'] = y_true
        dic['id_pred'] = y_pred
        dic['score'] = score
        return pd.DataFrame(dic) # , Classification_report, Confusion_matrix
    
    
    def test_intruder_detection(self):
        val_result_df = self.test(os.path.join(self.split_path, 'val'))
        val_report, val_conf_mat = self.confusion_Matrix(val_result_df['id_true'].values, val_result_df['id_pred'].values)
        val_all, val_right, val_wrong = result(val_result_df, val_report, val_conf_mat, self.result_path, 'val')
        
        self.intruders_score_threshold = (val_right['score_mean'] + val_right['score_std']).max()
        print('Intruder detection threshold: {}'.format(self.intruders_score_threshold))
        test_result_df = self.test(os.path.join(self.split_path, 'test'))
        test_result_df.loc[test_result_df['score'] > self.intruders_score_threshold, 'id_pred'] = 100
        test_report, test_conf_mat = self.confusion_Matrix(test_result_df['id_true'].values, test_result_df['id_pred'].values)
        test_all, test_right, test_wrong = result(test_result_df, test_report, test_conf_mat, self.result_path, 'test')
        
        # Saving some additional Information in the result folder
        text_file = os.path.join(self.result_path, 'details.txt')
        with open(text_file, 'w') as file:
            file.write('min_face_images: {}\n'.format(self.min_face_images))
            file.write('selected_person: {}\n'.format(list(self.selected_person_list)))
            file.write('Intruder: {}\n'.format(list(self.intruder_list)))
            file.write('Intruder min score: {}\n'.format(self.intruders_score_threshold))
        print('Results have been saved in {}'.format(self.result_path))
    
    
    def confusion_Matrix(self, y_true, y_pred):
        print("Classification Report:")
        Classification_report = classification_report(y_true, y_pred)
        print(Classification_report)
        print("\nConfusion Matrix:")
        Confusion_matrix = confusion_matrix(y_true, y_pred)
        print(Confusion_matrix)
        return Classification_report, Confusion_matrix
        
    
    def recognizePerson(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.__preprocess_image__(gray)
        label, score = self.model.predict(gray)
        return label, score