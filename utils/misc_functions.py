# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:27:12 2025
@author: Murad
SISLab, USF
mmurad@usf.edu
"""
import cv2
import os
import sys
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(data_dir, new_data_dir, selected_person_list, intuder_list, test_ratio = 0.2, val_ratio = 0.2, seed = 42, delete_dir = True):
    random.seed(seed)
    train_dir = os.path.join(new_data_dir, "train")
    test_dir = os.path.join(new_data_dir, "test")
    val_dir = os.path.join(new_data_dir, "val")
    
    if os.path.isdir(train_dir) and delete_dir:
        shutil.rmtree(train_dir)
    if os.path.isdir(test_dir) and delete_dir:
        shutil.rmtree(test_dir)
    if os.path.isdir(val_dir) and delete_dir:
        shutil.rmtree(val_dir)        
    
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok = True)
    
    for person_no in selected_person_list:
        person = str(person_no)
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        
        images = [img for img in os.listdir(person_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)
        train_ratio = 1 - (test_ratio + val_ratio)
        
        train_images, temp_images = train_test_split(images, test_size = (test_ratio + val_ratio), train_size = train_ratio, random_state = seed, shuffle = True)
        val_images, test_images = train_test_split(temp_images, test_size = test_ratio / (test_ratio + val_ratio), train_size = val_ratio / (test_ratio + val_ratio), random_state = seed, shuffle = True)

        for img in train_images:
            originial_image = os.path.join(person_dir, img)
            new_folder = os.path.join(train_dir, person)
            os.makedirs(new_folder, exist_ok=True)
            shutil.copy(originial_image, os.path.join(new_folder, img))
        
        for img in test_images:
            originial_image = os.path.join(person_dir, img)
            new_folder = os.path.join(test_dir, person)
            os.makedirs(new_folder, exist_ok=True)
            shutil.copy(originial_image, os.path.join(new_folder, img))
        
        for img in val_images:
            originial_image = os.path.join(person_dir, img)
            new_folder = os.path.join(val_dir, person)
            os.makedirs(new_folder, exist_ok=True)
            shutil.copy(originial_image, os.path.join(new_folder, img))
    
    ####################### Intruders #######################
    for person_no in intuder_list:
        person = str(person_no)
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        
        images = [img for img in os.listdir(person_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)

        for img in images:
            originial_image = os.path.join(person_dir, img)
            new_folder = os.path.join(test_dir, '100')
            os.makedirs(new_folder, exist_ok=True)
            shutil.copy(originial_image, os.path.join(new_folder, img))
            
    print("Split data into train and test.")

def result_for_intruders(result, path, text):
    intruders_df = {'pred_id': [], 'score_mean': [], 'score_std': [], 'score_min': [], 'score_max': []}
    for pred_id, (group) in result.groupby(['id_pred']):
        intruders_df['pred_id'].append(pred_id)
        intruders_df['score_mean'].append(group['score'].mean())
        intruders_df['score_std'].append(group['score'].std())
        intruders_df['score_min'].append(group['score'].min())
        intruders_df['score_max'].append(group['score'].max())

    intruders_df = pd.DataFrame(intruders_df)    
    os.makedirs(path, exist_ok = True)
    plot_table(intruders_df, 'Intruders_predicted_to_other_classes', path = os.path.join(path, '{}_Intruders_FN.png'.format(text)))
    return intruders_df

def result(result, class_report, conf_matrix, path, text):
    unique_person = np.sort(result['id_true'].unique())
    wrong_info = {'id': [], 'score_mean': [], 'score_std': [], 'score_min': [], 'score_max': []}
    right_info = {'id': [], 'score_mean': [], 'score_std': [], 'score_min': [], 'score_max': []}
    all_info = {'id': [], 'score_mean': [], 'score_std': [], 'score_min': [], 'score_max': []}
    for true_id, (group) in result.groupby(['id_true']):
        group_wrong = group[group['id_pred'] != group['id_true']] # False negative class
        group_right = group[group['id_pred'] == group['id_true']] # True Positive class
        
        wrong_info['id'].append(true_id)
        wrong_info['score_mean'].append(group_wrong['score'].mean())
        wrong_info['score_std'].append(group_wrong['score'].std())
        wrong_info['score_min'].append(group_wrong['score'].min())
        wrong_info['score_max'].append(group_wrong['score'].max())
        
        right_info['id'].append(true_id)
        right_info['score_mean'].append(group_right['score'].mean())
        right_info['score_std'].append(group_right['score'].std())
        right_info['score_min'].append(group_right['score'].min())
        right_info['score_max'].append(group_right['score'].max())
        
        all_info['id'].append(true_id)
        all_info['score_mean'].append(group['score'].mean())
        all_info['score_std'].append(group['score'].std())
        all_info['score_min'].append(group['score'].min())
        all_info['score_max'].append(group['score'].max())
    wrong_info = pd.DataFrame(wrong_info)
    right_info = pd.DataFrame(right_info)
    all_info = pd.DataFrame(all_info)
    
    os.makedirs(path, exist_ok = True)
    plot_table(all_info, text + '_all', path = os.path.join(path, '{}_alll.png'.format(text)))
    plot_table(right_info, text + '_TruePositiveClass', path = os.path.join(path, '{}_right.png'.format(text)))
    plot_table(wrong_info, text + '_FalseNegativeClass', path = os.path.join(path, '{}_wrong.png'.format(text)))
    
    ##########
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    ax.text(0.5, 0.5, class_report, fontsize=10, ha='center', va='center', family='monospace')
    output_report_path = os.path.join(path, '{}_classification_report.png'.format(text))
    plt.savefig(output_report_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save confusion matrix as an image
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2"],
                yticklabels=["Class 0", "Class 1", "Class 2"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    output_matrix_path =  os.path.join(path, '{}_confusion_matrix.png'.format(text))
    plt.savefig(output_matrix_path, bbox_inches='tight', dpi=300)
    plt.close()
    return all_info, right_info, wrong_info

def plot_table(df, title, path = []):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Turn off the axis
    table = plt.table(cellText=df.round(2).values,
                      colLabels=df.columns,
                      loc='center',
                      cellLoc='center')

    table.auto_set_font_size(False)    
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
