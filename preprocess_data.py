from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import argparse



def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour=6)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    y = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, y

def generate_train_val_test(args):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(args.traffic_df_filename)
    data_seq=np.expand_dims(data_seq,axis=2)
    print(data_seq.shape)
    
   
    num_of_weeks, num_of_days,num_of_hours,num_for_predict,points_per_hour=args.num_of_weeks,args.num_of_days,args.num_of_hours, args.num_for_predict,args.points_per_hour
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, y = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0)  #(1,T,N,F)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0)
            sample.append(hour_sample)

        y = np.expand_dims(y, axis=0)  # (1,T,N,F)
        sample.append(y)

        # time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        # sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,Tw,N,F),(1,Td,N,F),(1,Th,N,F),(1,Tpre,N),(1,1)]
        
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]
      # [(B,Tw,N,F),(B,Td,N,F),(B,Th,N,F),(B,Tpre,N),(B,1)]
    
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    x_train = np.concatenate(training_set[0:2], axis=1)  # (B,T',N,F), concat multiple time series segments (for week, day, hour) together
    x_val = np.concatenate(validation_set[0:2], axis=1)
    x_test = np.concatenate(testing_set[0:2], axis=1)
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    # x_train = training_set[-2] # (B,T',N,F), concat multiple time series segments (for week, day, hour) together
    # x_val = validation_set[-2]
    # x_test = testing_set[-2]
    # print(x_train.shape)
    # print(x_val.shape)
    # print(x_test.shape)

    y_train = training_set[-1]  # (B,T,N)
    y_val = validation_set[-1]
    y_test = testing_set[-1]

    
    
    all_data={
        'train':{
            'x':x_train,
            'y':y_train,
           
        },
        'val':{
            'x':x_val,
            'y':y_val,
           
        },
        'test':{
            'x':x_test,
            'y':y_test,
           
        }
    }

    for cat in ["train","val","test"]:
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz"%cat),
            x=all_data[cat]['x'],  
            y=all_data[cat]['y'], 
          
        )
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="HiCo-STEP_v1/data/XiAn_City", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="HiCo-STEP_v1/data/xianCO_12.npy", help="Raw traffic readings.",)
    parser.add_argument("--dow", action='store_true',)
    parser.add_argument("--num_of_weeks", type=int, default=0,help="none")
    parser.add_argument("--num_of_days", type=int, default=2)
    parser.add_argument("--num_of_hours", type=int, default=1)
    parser.add_argument("--num_for_predict", type=int, default=12)
    parser.add_argument("--points_per_hour", type=int, default=12)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input('%s exists. Do you want to overwrite it? (y/n)'%args.output_dir)).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
