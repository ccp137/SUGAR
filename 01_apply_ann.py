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

Version 1.1

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
import os
from obspy.core import UTCDateTime
import numpy as np  
import pandas as pd
import h5py
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from obspy import read
import scipy
from scipy.signal import butter, lfilter
from obspy.geodetics.base import gps2dist_azimuth
import sys
import argparse
# ========================================================
def butter_bandpass(low_freq, high_freq, sampling_rate, order=5):
    nyquist_freq = 0.5 * sampling_rate
    low_f = low_freq / nyquist_freq
    high_f = high_freq / nyquist_freq
    b, a = butter(order, [low_f, high_f], btype='band')
    return b, a
# ========================================================
def butter_bandpass_filter(data, low_freq, high_freq, sampling_rate, order=5):
    b, a = butter_bandpass(low_freq, high_freq, sampling_rate, order=order)
    y = lfilter(b, a, data)
    return y
# ========================================================
def cal_skewness(x):
    x = pd.Series(x)
    return pd.Series.skew(x)
# ========================================================
def cal_features(data):
    x = data
    features = {}
    
    abs_energy = np.dot(x, x)
    features['abs_energy'] = abs_energy
    
    abs_sum_of_changes = np.sum(np.abs(np.diff(x)))
    features['abs_sum_of_changes'] = abs_sum_of_changes
    
    kurtosis = scipy.stats.kurtosis(x)
    features['kurtosis'] = kurtosis
    
    length = len(x)
    features['length'] = length
    
    maximum = np.max(x)
    features['maximum'] = maximum
    
    mean = np.mean(x)
    features['mean'] = mean
    
    mean_abs_change = np.mean(np.abs(np.diff(x)))
    features['mean_abs_change'] = mean_abs_change
    
    mean_change = np.mean(np.diff(x))
    features['mean_change'] = mean_change
    
    median = np.median(x)
    features['median'] = median
    
    minimum = np.min(x)
    features['minimum'] = minimum
    
    skewness = cal_skewness(x)
    features['skewness'] = skewness
    
    std = np.std(x)
    features['std'] = std
    
    sum_values = np.sum(x)
    features['sum_values'] = sum_values

    percent_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    quantile_list = np.percentile(x, percent_list)

    quantile = quantile_list[-1] - quantile_list[0]
    features['quantile_90'] = quantile
    
    quantile = quantile_list[-2] - quantile_list[1]
    features['quantile_80'] = quantile
    
    quantile = quantile_list[-3] - quantile_list[2]
    features['quantile_70'] = quantile

    quantile = quantile_list[-4] - quantile_list[3]
    features['quantile_60'] = quantile

    quantile = quantile_list[-5] - quantile_list[4]
    features['quantile_50'] = quantile

    quantile = quantile_list[-6] - quantile_list[5]
    features['quantile_40'] = quantile

    quantile = quantile_list[-7] - quantile_list[6]
    features['quantile_30'] = quantile

    quantile = quantile_list[-8] - quantile_list[7]
    features['quantile_20'] = quantile

    quantile = quantile_list[-9] - quantile_list[8]
    features['quantile_10'] = quantile
    
    return features
# ========================================================
if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Predict quality probability using a pre-trained model')
    parser.add_argument('--filelist_name', default='filelist.txt', help='Path to a file contains filenames of SAC files that need to be graded (default: filelist.txt)')
    parser.add_argument('--data_dir', default='./test_data/', help='Folder where the surface-wave seismograms are stored (default: ./test_data/).')
    parser.add_argument('--model_name', default='./trained_model/ann_202110131500.hdf5', help='Machine learning model to be used (default: ./trained_model/ann_202010210900.hdf5)')
    parser.add_argument('--output_dir', default='./', help='Folder to save output (default: ./)')
    parser.add_argument('--batch_size', default=5, help='Batch size.', type=int)
    args = parser.parse_args()
    model_name = args.model_name
    data_dir = args.data_dir
    filelist_fname = args.filelist_name
    output_dir = args.output_dir
    batch_size = args.batch_size
    # ---------------------------------------------------------
    # load model and scaler
    classifier_ann = load_model(model_name)
    classifier = classifier_ann
    #
    sc = StandardScaler()
    h5_fid = h5py.File(model_name, 'r')
    scaler_group = h5_fid['StandardScaler']
    sc.mean_ = scaler_group['mean_'][...]
    sc.scale_ = scaler_group['scale_'][...]
    sc.var_ = scaler_group['var_'][...]
    sc.n_features_in_ = len(scaler_group['var_'][...])#scaler_group.attrs['n_features_in_']
    h5_fid.close()
    # ---------------------------------------------------------
    # read in a list of filenames
    file_list_df = pd.read_csv(filelist_fname, names=['fname'])
    file_list = file_list_df['fname'].values
    # ---------------------------------------------------------
    group_vel_min = 2.5
    group_vel_max = 5.0
    low_freq=1./60
    high_freq=1./30
    nwindow = 10
    nfeature = 301
    grade = 0 # default grade
    # process 
    batch_fname_list = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    #
    for ibatch in range(len(batch_fname_list)):
        feature_list = []
        used_files = []
        bad_files = []
        divide_list = ['abs_energy', 'skewness', 'kurtosis', 'maximum', 'mean', 'median', 'minimum', 'std', 
                           'quantile_10', 'quantile_20', 'quantile_30', 'quantile_40', 'quantile_50',
                           'quantile_60', 'quantile_70', 'quantile_80', 'quantile_90']
        batch_files = batch_fname_list[ibatch]
        for afname in batch_files:
            afile = data_dir.strip()+'/'+afname.strip()
            if os.path.exists(afile):
                st = read(afile)
                tr = st[0]
                sampling_rate = tr.stats.sampling_rate
                starttime = tr.stats.starttime
                raw_data = tr.data
                if len(raw_data) > 0:
                    trace_id = afile
                    data = butter_bandpass_filter(raw_data, low_freq, high_freq, sampling_rate, order=4)
                    meta = tr.stats.sac
                    dist, az, baz = gps2dist_azimuth(meta['stla'], meta['stlo'], meta['evla'], meta['evlo'])
                    dist = dist * 0.001 # meter to km
                    t_max = dist / group_vel_min
                    t_min = dist / group_vel_max
                    origin_time = UTCDateTime(year=meta['nzyear'], julday=meta['nzjday'], hour=meta['nzhour'],\
                                             minute=meta['nzmin'], second=meta['nzsec'], microsecond=meta['nzmsec'])
                    surface_wave_time_start = origin_time + t_min
                    surface_wave_time_end = origin_time + t_max
                    surface_window_length = int((surface_wave_time_end - surface_wave_time_start) * sampling_rate)
                    sample_window_length = surface_window_length*1./nwindow
                    if sample_window_length > 10:
                        #
                        surface_start_index = int((surface_wave_time_start - starttime) * sampling_rate)
                        surface_end_index = int((surface_wave_time_end - starttime) * sampling_rate)
                        #
                        expected_window_length = (dist / group_vel_min - dist /group_vel_max) * sampling_rate
                        surface_wave_data = data[surface_start_index:surface_end_index]
                        temp_start = max(0,surface_start_index-surface_window_length)
                        before_data = data[temp_start:surface_start_index]
                        if len(surface_wave_data) > surface_window_length*0.99 and len(before_data) > surface_window_length*0.5:
                            # compute features for the entire trace
                            features_merged = {}
                            features_before = cal_features(before_data)
                            for akey in features_before:
                                features_merged['before_'+akey] = features_before[akey]
                            features_surface_wave = cal_features(surface_wave_data)
                            for akey in features_surface_wave:
                                features_merged['surface_'+akey] = features_surface_wave[akey]
                            #
                            for iwindow in range(nwindow):
                                istart = surface_start_index + int(iwindow*sample_window_length)
                                iend = min(surface_start_index + int((iwindow+1)*sample_window_length), surface_end_index)
                                features_sample = cal_features(data[istart:iend])
                                for akey in features_sample:
                                    features_merged['sub'+str(iwindow).zfill(3)+'_'+akey] = features_sample[akey] / (features_merged['surface_'+akey] + 1e-19)
                            #
                            energy_list = []
                            for iwindow in range(nwindow):
                                energy_list.append(features_merged['sub'+str(iwindow).zfill(3)+'_abs_energy'])
                            min_index = np.argmin(energy_list)
                            max_index = np.argmax(energy_list)
                            for idiv in range(len(divide_list)):
                                div_key = divide_list[idiv]
                                features_merged['max_over_min_'+div_key] = features_merged['sub'+str(max_index).zfill(3)+'_'+div_key] / (features_merged['sub'+str(min_index).zfill(3)+'_'+div_key] + 1e-19)
                            for idiv in range(len(divide_list)):
                                div_key = divide_list[idiv]
                                features_merged['surface_over_before_'+div_key] = features_merged['surface_'+div_key] / (features_merged['before_'+div_key] + 1e-19)
                            #
                            features_merged['mag'] = float(meta['mag'])
                            features_merged['evt_depth'] = float(meta['evdp'])
                            features_merged['az'] = az
                            features_merged['distance'] = dist
                            features_merged['grade'] = grade
                            features_merged['id'] = trace_id
                            feature_list.append(features_merged)
                            used_files.append(afname)
                        else:
                            bad_files.append(afname)
                    else:
                        bad_files.append(afname)
                else:
                    bad_files.append(afname)
            else:
                print("Can't find ", afile)
                sys.exit()
        #
        attrs_list = ['id', 'grade']
        x_list = []
        for a_feature in feature_list:
            trace_id = a_feature['id']
            data_list = []
            sorted_keys = np.sort(list(a_feature.keys())) 
            for akey in sorted_keys:
                if akey not in attrs_list:
                    data_list.append(a_feature[akey])
            x_list.append(np.array(data_list))
        #
        if len(x_list) > 0:
            x_deploy = sc.transform(x_list)
            y_deploy = classifier.predict(x_deploy)
            #y_label = (y_deploy >= 0.5).astype(int).flatten()
            #
            fid = open(output_dir+'/predicted_prob.txt', 'a')
            for i in range(len(feature_list)):
                fid.write('{0}, {1:10.4f} \n'.format(used_files[i].strip(), y_deploy[i][0]))
            fid.close()
        #
        if len(bad_files) > 0:
            fid = open(output_dir+'/files_not_graded.txt', 'a')
            for i in range(len(bad_files)):
                fid.write('{0} \n'.format(bad_files[i].strip()))
            fid.close()
