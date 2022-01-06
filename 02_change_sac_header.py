#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

SUrface-wave Grader with ARtificial intelligence (SUGAR)
Copyright (C) 2022  Chengping Chai, Jingyi Luo, and Monica Maceira

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

version 1.0

Developed by Chengping Chai (chaic@ornl.gov) in collaboration with 
Jonas Kintner, K. Michael Cleveland, Jingyi Luo, Monica Maceira, Charles J. Ammon

This package (SUrface-wave Grader with ARtificial intelligence or SUGAR) automatically assigns 
a quality score to surface-wave seismograms (SAC format) using a trained artificial neural network model (included). 
Specifically, the python script 01_apply_ann.py calculates probability scores for a list of SAC files. 
You may consider seismograms with probability scores larger than 0.5 as acceptable data. 
Note no scores will be given to seismograms that do not pass an initial check (e.g., insufficient number of data points). 
See the following paper for more details. 

If you use this package for your research, please consider cite:

Chai, C., Kintner, J., Cleveland, K. M., Luo, J., Maceira, M., & Ammon, C. J. (2021). 
    Automatic Waveform Quality Control for Surface Waves Using Machine Learning. 
    https://doi.org/10.1002/essoar.10507941.3

'''
import pandas as pd
from obspy import read
import argparse
# ========================================================
def probability_to_grade(prob, threshold, tol=1e-6):
    if threshold >= 0.5:
        if prob >= threshold:
            grade = 2
        elif prob < threshold and prob >= 0.5:
            grade = 1
        else:
            grade = 0
    else:
        print('Please use a threshold larger than or equal to 0.5!')
    return grade

# ========================================================
if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Change SAC header iqual')
    parser.add_argument('--threshold', default=0.5, type=float, help='Probability threshold to compute quality grades (default: 0.5)')
    parser.add_argument('--data_dir', default='./test_data/', help='Data directory (default: ./test_data/)')
    parser.add_argument('--pred_file', default='./predicted_prob.txt', help='Path to the file that contains the predicted probabilities (default: ./predicted_prob.txt)')
    args = parser.parse_args()
    #
    threshold = args.threshold
    data_dir = args.data_dir
    pred_file = args.pred_file
    #
    pred_prob_df = pd.read_csv(pred_file, names=['fname', 'prob'])
    #
    for i in range(len(pred_prob_df)):
        fname = pred_prob_df['fname'][i]
        st = read(data_dir+'/'+fname)
        prob = float(pred_prob_df['prob'][i])
        grade = probability_to_grade(prob, threshold=threshold)
        st[0].stats.sac['iqual'] = grade
        st.write(data_dir+'/'+fname)