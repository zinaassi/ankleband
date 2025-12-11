import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import convolve1d
from scipy import signal
from torch.utils.data import Dataset
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import swifter

class DataManagement(object):

    def __init__(self, cfg):
        super(DataManagement, self).__init__()

        # init parameters
        self.cfg = cfg

        # load and preprocess data
        self.load_data()

        # apply low-pass filter if enabled (before dilution and concatenation)
        if hasattr(self.cfg.data, 'apply_filter') and self.cfg.data.apply_filter:
            self.apply_lowpass_filter()

        self.dilute_data()

        # prepare concatenated samples
        self.remove_irrelevant_data()
        self.concatenate_samples()

        # prepare inputs and outputs
        self.prepare_inputs()
        self.prepare_output_labels()

        if self.cfg.training.weighted_sampling:
            self.prepare_sample_weighting()

        if self.cfg.data.dtw:
            self.prepare_dtw_features()

        if not self.cfg.data.single_test:
            print('Total samples for train: {}.\nTotal samples for test: {}.'.format(len(self.train_df),len(self.test_df)))
        else:
            print('Total samples for single test: {}.'.format(len(self.test_df)))
        
        # just a sanity check
        self.remove_nan_values()
        
        # prepare matrices for training
        if cfg.model.type == 'classic':
            self.prepare_matrices()

    # load all the h5 files for training and testing
    def load_data(self):

        print('Loading dataset...')

        # load train h5 files
        train_dfs, test_dfs = [], []
        for h5_file in [os.path.join(self.cfg.data.path, x) for x in self.cfg.data.train_files]:
            
            # check if file exist
            if os.path.isfile(h5_file):
                train_dfs.append(pd.read_hdf(path_or_buf=h5_file,key='df'))
                train_dfs[-1]['sample_index'] = train_dfs[-1].index
                train_dfs[-1]['subject'] = int(h5_file.split('/')[-1].split('_')[0].split('ID')[1])
            else:
                print(f'{h5_file} was not found!')

        # load test hf files. make sure to load only one of them if asked
        for h5_file in [os.path.join(self.cfg.data.path, x) for x in self.cfg.data.test_files]:
            
            # check if file exist           
            if os.path.isfile(h5_file):
                test_dfs.append(pd.read_hdf(path_or_buf=h5_file,key='df'))
                test_dfs[-1]['sample_index'] = test_dfs[-1].index
                test_dfs[-1]['subject'] = int(h5_file.split('/')[-1].split('_')[0].split('ID')[1])
            else:
                print(f'{h5_file} was not found!')

        # share test sessions with train sessions by moving part of each dataframe to train if asked
        test_dfs_updated = []
        if self.cfg.data.share_train > 0.0:

            data_freq = 200
            train_mins = 24
            test_mins = 6

            # define k-fold
            if self.cfg.data.kfold is not None:
                k_fold_size = 1 - self.cfg.data.share_train

                for i in range(len(test_dfs)):

                    # iteratively take a fold to the test data by iterating over 30 seconds
                    j = 0
                    while j < len(test_dfs[i]):

                        fold_start_idx = j + data_freq * train_mins
                        fold_end_idx = j + data_freq * train_mins + data_freq * test_mins

                        # separate training samples and add them to the list
                        if fold_start_idx > len(test_dfs[i]):
                            fold_start_idx = len(test_dfs[i])
                        train_df_fold = test_dfs[i].iloc[j:fold_start_idx]
                        if len(train_df_fold) > 0:
                            train_dfs.append(train_df_fold)

                        
                        if fold_start_idx < len(test_dfs[i]):
                            if fold_end_idx > len(test_dfs[i]):
                                fold_end_idx = len(test_dfs[i])
                            test_dfs_updated.append(test_dfs[i].iloc[fold_start_idx:fold_end_idx])
                        j = fold_end_idx

                    # define fold start and end indices
                    # fold_start_idx = int(self.cfg.data.kfold * (k_fold_size * len(test_dfs[i])))
                    # fold_end_idx = int((self.cfg.data.kfold + 1) * (k_fold_size * len(test_dfs[i])))

                    # # separate training samples and add them to the list
                    # train_df_bfold = test_dfs[i].iloc[:fold_start_idx]
                    # train_df_afold = test_dfs[i].iloc[fold_end_idx:]
                    # if len(train_df_bfold) > 0:
                    #     train_dfs.append(train_df_bfold)
                    # if len(train_df_afold) > 0:
                    #     train_dfs.append(train_df_afold)

                    # test_dfs[i] = test_dfs[i].iloc[fold_start_idx:fold_end_idx]

                test_dfs = test_dfs_updated
            else:
                for i in range(len(test_dfs)):
                    train_dfs.append(test_dfs[i].iloc[:int(len(test_dfs[i]) * self.cfg.data.share_train)])
                    test_dfs[i] = test_dfs[i].iloc[int(len(test_dfs[i]) * self.cfg.data.share_train):]
        # set aside dataframes related to the required subject for test 
        elif self.cfg.data.leave_subject_out is not None:
            test_dfs = [x for x in train_dfs if x.iloc[0].subject == self.cfg.data.leave_subject_out]
            train_dfs = [x for x in train_dfs if x.iloc[0].subject != self.cfg.data.leave_subject_out]

            if self.cfg.data.force_num_subjects_train is not None:
                # get the indeices of available subjects in the training dataframe
                train_subjects = np.unique([x.iloc[0].subject for x in train_dfs])

                # randomly select subjects to keep and keep only those in the training dataframes
                train_subjects = np.random.choice(train_subjects, self.cfg.data.force_num_subjects_train, replace=False)
                train_dfs = [x for x in train_dfs if x.iloc[0].subject in train_subjects]

        # concatenate them into single dataframes
        if not self.cfg.data.single_test:
            self.train_df = pd.concat(train_dfs)
        else:
            self.train_df = pd.DataFrame()
        self.test_df = pd.concat(test_dfs)

    # dilute data by sekecting only samples after strides
    def dilute_data(self):
        if self.cfg.data.stride > 1:
            if not self.cfg.data.single_test:
                self.train_df = self.train_df.iloc[::self.cfg.data.stride,:]
            self.test_df = self.test_df.iloc[::self.cfg.data.stride,:]

    # apply low-pass Butterworth filter to reduce sensor noise
    def apply_lowpass_filter(self):
        """
        Apply Butterworth low-pass filter to reduce sensor noise.
        Filters raw IMU data before normalization and windowing.
        Based on the paper's approach to handling noisy IMU data.
        """

        print('Applying low-pass filter to sensor data...')

        # Filter parameters
        sampling_rate = 200  # Hz (from dataset collection at 200 Hz)
        cutoff_frequency = self.cfg.data.filter_cutoff  # From config
        filter_order = self.cfg.data.filter_order  # From config

        print(f'Filter settings: Cutoff={cutoff_frequency}Hz, Order={filter_order}, Sampling={sampling_rate}Hz')

        # Design Butterworth low-pass filter
        nyquist_freq = sampling_rate / 2.0  # Nyquist frequency = 100 Hz
        normalized_cutoff = cutoff_frequency / nyquist_freq  # Normalize to 0-1 range

        # Create filter coefficients using Second-Order Sections (SOS) - best practice for numerical stability
        sos = signal.butter(filter_order, normalized_cutoff, btype='low', output='sos')

        # Sensor columns to filter (6-axis IMU: 3 accelerometer + 3 gyroscope)
        sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        # 9 8 8 6 6 5 4 3 2 1 0
        # Apply zero-phase filtering to training data
        if not self.cfg.data.single_test:
            print('Filtering training data...')
            for col in sensor_columns:
                # sosfilt = causal filtering (forward-only filter, compatible with real-time ESP32)
                self.train_df[col] = signal.sosfilt(sos, self.train_df[col].values)

        # Apply causal filtering to test data
        print('Filtering test data...')
        for col in sensor_columns:
            self.test_df[col] = signal.sosfilt(sos, self.test_df[col].values)

        print('Low-pass filtering complete.')

    # remove columns irrelevant for training
    def remove_irrelevant_data(self):

        print('Removing irrelevant data...')

        self.irrelavant_columns = [x for x in self.test_df.columns.tolist() if 'leg_skeleton' in x] + \
                                  [x for x in self.test_df.columns.tolist() if 'time_' in x] + ['subject']
        if not self.cfg.data.single_test:
            self.train_df.drop(columns=self.irrelavant_columns, inplace=True)
        self.test_df.drop(columns=self.irrelavant_columns, inplace=True)

    # append images and labels to create stacked inputs and outputs
    def concatenate_samples(self):

        print('Concatenating samples...')

        # prepare columns for concatenation
        self.labels_names = ['label']
        self.labels_names_stacked = [x+'s' for x in self.labels_names]
        if not self.cfg.data.single_test:
            self.train_df[self.labels_names_stacked] = self.train_df[self.labels_names].map(lambda x: [x])
        self.test_df[self.labels_names_stacked] = self.test_df[self.labels_names].map(lambda x: [x])

        # create new column for stacked inputs and convert to float32
        # self.inputs_names = ['dx', 'dy', 'dz', 'dth', 'dph', 'dps']
        self.inputs_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.inputs_names_stacked = [x+'s' for x in self.inputs_names]
        if not self.cfg.data.single_test:
            self.train_df[self.inputs_names_stacked] = self.train_df[self.inputs_names].map(lambda x: [x])
        self.test_df[self.inputs_names_stacked] = self.test_df[self.inputs_names].map(lambda x: [x])

        # backup dataframes
        if len(self.labels_names_stacked) > 0:
            if not self.cfg.data.single_test:
                train_df_temp = self.train_df.copy()
            test_df_temp = self.test_df.copy()

        # shift by the step size to append samples together
        for i in range(self.cfg.data.step,self.cfg.data.append*self.cfg.data.step,self.cfg.data.step):
            # append inputs
            if not self.cfg.data.single_test:
                self.train_df[self.inputs_names_stacked] = train_df_temp.shift(i)[self.inputs_names_stacked] + self.train_df[self.inputs_names_stacked]
            self.test_df[self.inputs_names_stacked] = test_df_temp.shift(i)[self.inputs_names_stacked] + self.test_df[self.inputs_names_stacked]

            # append labels
            if len(self.labels_names_stacked) > 0:
                if not self.cfg.data.single_test:
                    self.train_df[self.labels_names_stacked] = train_df_temp.shift(i)[self.labels_names_stacked] + self.train_df[self.labels_names_stacked]
                self.test_df[self.labels_names_stacked] = test_df_temp.shift(i)[self.labels_names_stacked] + self.test_df[self.labels_names_stacked]

        # drop rows with missing information
        if not self.cfg.data.single_test:
            self.train_df = self.train_df.iloc[self.cfg.data.append*self.cfg.data.step-1:]
        self.test_df = self.test_df.iloc[self.cfg.data.append*self.cfg.data.step-1:]

        # convert labels to numpy for future computations
        if not self.cfg.data.single_test:
            self.train_df[self.labels_names_stacked] = self.train_df[self.labels_names_stacked].map(lambda x: np.array(x))
            self.train_df[self.inputs_names_stacked] = self.train_df[self.inputs_names_stacked].map(lambda x: np.array(x, dtype=np.float32))
        self.test_df[self.labels_names_stacked] = self.test_df[self.labels_names_stacked].map(lambda x: np.array(x))
        self.test_df[self.inputs_names_stacked] = self.test_df[self.inputs_names_stacked].map(lambda x: np.array(x, dtype=np.float32))

    # prepare inputs for training
    def prepare_inputs(self):

        print('Normalizaing inputs...')

        # normalize inputs according to their reasonable max value
        MAX_ACC = 10.0
        MAX_GYRO = 2.0
        if not self.cfg.data.single_test:
            self.train_df[self.inputs_names_stacked[:3]] = self.train_df[self.inputs_names_stacked[:3]].map(lambda x: x / MAX_ACC) # Accelerometer values
            self.train_df[self.inputs_names_stacked[3:]] = self.train_df[self.inputs_names_stacked[3:]].map(lambda x: x / MAX_GYRO) # Gyro values
        self.test_df[self.inputs_names_stacked[:3]] = self.test_df[self.inputs_names_stacked[:3]].map(lambda x: x / MAX_ACC) # Accelerometer values
        self.test_df[self.inputs_names_stacked[3:]] = self.test_df[self.inputs_names_stacked[3:]].map(lambda x: x / MAX_GYRO) # Gyro values

        # concatenate all inputs into one
        if not self.cfg.data.single_test:
            self.train_df['input_vec'] = self.train_df.apply(lambda x: np.concatenate([x[y][None,:] for y in self.inputs_names_stacked]), axis=1)
        self.test_df['input_vec'] = self.test_df.apply(lambda x: np.concatenate([x[y][None,:] for y in self.inputs_names_stacked]), axis=1)

    # prepare labels for classification
    def prepare_output_labels(self):

        if not self.cfg.data.single_test:
            self.train_df['label_max'] = self.train_df['labels'].apply(max)
            self.train_df['label_percentage'] = self.train_df.apply(lambda x: ((x['labels'] / x['label_max']).sum() / self.cfg.data.append) if x['label_max'] > 0 else 0, axis=1)
            self.train_df['label_idx'] = self.train_df.apply(lambda x: x['label_max'] if x['label_percentage'] >= self.cfg.data.label_percentage else 0, axis=1)
        self.test_df['label_max'] = self.test_df['labels'].apply(max)
        self.test_df['label_percentage'] = self.test_df.apply(lambda x: ((x['labels'] / x['label_max']).sum() / self.cfg.data.append) if x['label_max'] > 0 else 0, axis=1)
        self.test_df['label_idx'] = self.test_df.apply(lambda x: x['label_max'] if x['label_percentage'] >= self.cfg.data.label_percentage else 0, axis=1)

        # convert to binary label
        if self.cfg.data.classes == 1:
            if not self.cfg.data.single_test:
                self.train_df['label_idx'] = self.train_df['label_idx'].apply(lambda x: 1 if x > 0 else 0)
            self.test_df['label_idx'] = self.test_df['label_idx'].apply(lambda x: 1 if x > 0 else 0)

    # prepare features using 1D convolutions
    def prepare_sample_weighting(self):
        
        label_amounts = self.train_df['label_idx'].value_counts()
        if not self.cfg.data.single_test:
            self.train_df['weight'] = self.train_df['label_idx'].apply(lambda x: 1/label_amounts[x])
        self.test_df['weight'] = self.test_df['label_idx'].apply(lambda x: 1/label_amounts[x])

    # prepare features using DTW
    def prepare_dtw_features(self):

        if self.cfg.data.single_test:
            print('ERROR! Cannot compute DTW without train data.')
            return
        
        print('Computing DTW features...')
        num_candidates = 10

        # start with sampling cadidates for each class
        self.dtw_candidates = {}
        for i in range(self.cfg.data.classes):
            print('class number: ', i)
            self.dtw_candidates[i] = self.train_df[self.train_df['label_idx'] == i].sample(n=num_candidates).input_vec

            # compute DTW features for each class
            self.train_df['dtw_candidate_temp'] = [self.dtw_candidates[i].iloc[0]] * len(self.train_df)
            print('train set:')
            for j in range(num_candidates):
                print('candidate number: ', j)
                self.train_df[f'candidate_features_{j}'] = self.train_df.swifter.apply(lambda x: self.compute_dtw_for_class(x['input_vec'], x['dtw_candidate_temp']), axis=1)

            # create features for each class distances and drop unwanted columns
            self.train_df[f'dtw_features_{i}'] = self.train_df.swifter.apply(lambda x: np.array([x[[f'candidate_features_{j}' for j in range(num_candidates)]].mean(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].std(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].min(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].max()]), axis=1)
            self.train_df.drop(columns=[f'candidate_features_{j}' for j in range(num_candidates)] + ['dtw_candidate_temp'], inplace=True)

            # compute DTW features for each class
            self.test_df['dtw_candidate_temp'] = [self.dtw_candidates[i].iloc[0]] * len(self.test_df)
            print('test set:')
            for j in range(num_candidates):
                print('candidate number: ', j)
                self.test_df[f'candidate_features_{j}'] = self.test_df.swifter.apply(lambda x: self.compute_dtw_for_class(x['input_vec'], x['dtw_candidate_temp']), axis=1)

            # create features for each class distances and drop unwanted columns
            self.test_df[f'dtw_features_{i}'] = self.test_df.swifter.apply(lambda x: np.array([x[[f'candidate_features_{j}' for j in range(num_candidates)]].mean(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].std(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].min(),
                                                                                                 x[[f'candidate_features_{j}' for j in range(num_candidates)]].max()]), axis=1)
            self.test_df.drop(columns=[f'candidate_features_{j}' for j in range(num_candidates)] + ['dtw_candidate_temp'], inplace=True)

        # concatenate DTW features and drop unwanted columns
        self.train_df['input_vec'] = self.train_df.apply(lambda x: np.concatenate([x[f'dtw_features_{i}'] for i in range(self.cfg.data.classes)]), axis=1)
        self.test_df['input_vec'] = self.test_df.apply(lambda x: np.concatenate([x[f'dtw_features_{i}'] for i in range(self.cfg.data.classes)]), axis=1)
        self.train_df.drop(columns=[f'dtw_features_{i}' for i in range(self.cfg.data.classes)], inplace=True)
        self.test_df.drop(columns=[f'dtw_features_{i}' for i in range(self.cfg.data.classes)], inplace=True)

        # normalize DTW features
        max_val = self.train_df['input_vec'].apply(np.max).max()
        print('Max value for DTW features: ', max_val)
        self.train_df['input_vec'] = self.train_df['input_vec'].map(lambda x: x / max_val)
        self.test_df['input_vec'] = self.test_df['input_vec'].map(lambda x: x / max_val)

        print('DTW features ready.')

    # compute DTW distance between two samples
    def compute_dtw_for_class(self, input_vec1, input_vec2):
        
        candidate_distance = fastdtw(input_vec1, input_vec2, dist=euclidean)[0]
        return candidate_distance
    
    # remove nan values. do nothing if there are no nan values
    def remove_nan_values(self):
        
        if not self.cfg.data.single_test:
            if len(self.train_df) > len(self.train_df.dropna()):
                print('WARNING: removed {} lines of train set'.format(len(self.train_df) - len(self.train_df.dropna())))
                self.train_df = self.train_df.dropna()
        self.test_df = self.test_df.dropna()

    # prepare X and y for sklearn training
    def prepare_matrices(self):

        if not self.cfg.data.dtw:
            if not self.cfg.data.single_test:
                self.train_df['input_vec'] = self.train_df['input_vec'].apply(lambda x: x.flatten())
            self.test_df['input_vec'] = self.test_df['input_vec'].apply(lambda x: x.flatten())

        if not self.cfg.data.single_test:
            self.train_X = np.vstack(self.train_df['input_vec'].values)
            self.train_y = self.train_df['label_idx'].values
        self.test_X = np.vstack(self.test_df['input_vec'].values)
        self.test_y = self.test_df['label_idx'].values

def compute_labels_distance(sample_values, sec_sample_values, is_label_only=False):

    # compute distance between two samples
    label1 = sample_values['label']
    label2 = sec_sample_values['label']

    label_percentage_dist = abs(sample_values['label_percentage'] - sec_sample_values['label_percentage']) / 2
    if is_label_only:
        label_percentage_dist = 0

    if label1 != label2: # distance defined by both label type and percentage
        return 0.5 + label_percentage_dist
    else: # distance defined by label percentage only
        return label_percentage_dist

class TorchDatasetManagement(Dataset):
    def __init__(self, cfg, data_df, inputs_names_stacked, is_train=True):

        # init parameters
        self.cfg = cfg
        self.data_df = data_df
        self.inputs_names_stacked = inputs_names_stacked
        self.is_train = is_train

        if self.cfg.training.weighted_sampling:
            self.resampled_data_size = self.data_df['label_idx'].value_counts().min() * self.cfg.data.classes
            self.sample_df = self.data_df.sample(n=self.resampled_data_size, weights='weight')

    def resample_data(self):
        self.sample_df = self.data_df.sample(n=self.resampled_data_size, weights='weight')

    def __len__(self):
        if self.cfg.training.weighted_sampling:
            return self.resampled_data_size
        else:
            return len(self.data_df)

    def __getitem__(self, idx):

        # draw sample
        if self.cfg.training.weighted_sampling:
            sample_values = self.sample_df.iloc[idx]
        else:
            sample_values = self.data_df.iloc[idx]

        # convert input and label tensors to torch tensors
        inputs = torch.from_numpy(np.vstack(sample_values[self.inputs_names_stacked].values))
        labels = torch.tensor(np.array(sample_values['label_idx']))

        # augmentations for training - gaussian noise to input
        if self.is_train:
            inputs = inputs + (torch.randn(inputs.shape) * 0.025)

        if self.cfg.mode == 'distance':

            # get a random index
            rand_idx = idx
            while (rand_idx == idx):
                rand_idx = random.randint(0, self.__len__() - 1)

            # draw second sample
            if self.cfg.training.weighted_sampling:
                sec_sample_values = self.sample_df.iloc[rand_idx]
            else:
                sec_sample_values = self.data_df.iloc[rand_idx]

            # convert second input and label tensors to torch tensors
            sec_inputs = torch.from_numpy(np.vstack(sec_sample_values[self.inputs_names_stacked].values))
            sec_labels = torch.tensor(np.array(sec_sample_values['label_idx']))

            labels_distance = compute_labels_distance(sample_values, sec_sample_values)

            return inputs, sec_inputs, labels_distance
        elif self.cfg.mode == 'downsample':
            return inputs
        else: # self.cfg.mode == 'regular'
            return inputs, labels