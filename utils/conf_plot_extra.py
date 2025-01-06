# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:10:51 2024

@author: Murad
"""

# Re-import necessary libraries after reset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Re-create a random confusion matrix for the given 26 crew IDs
crew_ids = [1, 2, 3, 4, 5, 7, 12, 19, 22]#, 'Intruder']
num_classes = conf_mat.shape[0]


# Plot confusion matrix
plt.figure() #(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=crew_ids, yticklabels=crew_ids)
# plt.title("Confusion Matrix for Crew IDs")
plt.xlabel("Predicted Crew ID")
plt.ylabel("True Crew ID")
plt.show()
