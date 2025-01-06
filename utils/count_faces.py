# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:18:21 2024

@author: Murad
"""

import os
import pandas as pd

def countfaces(data_dir):
    person = os.listdir(data_dir)
    dic = {'person': [], 'number': []}
    
    for p in person:
        person_dir = os.path.join(data_dir, p)
        person_num = len(os.listdir(person_dir))
        
        dic['person'].append(int(p))
        dic['number'].append(person_num)
        
    dic = pd.DataFrame(dic)
    
    return dic