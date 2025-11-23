import os
import json
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn

class ConfigManager(object):

    def __init__(self, json_name, random_seed=None):
        super(ConfigManager, self).__init__()

        # load json file
        self.json_path = os.path.join(os.getcwd(), json_name)
        with open(self.json_path) as f:
            cfg_dict = json.load(f)

        # add main properties
        self.output_dir = cfg_dict['OUTPUT_DIR']
        self.store_csv = cfg_dict['STORE_CSV']
        if 'IS_TRAINING' in cfg_dict:
            self.is_training = cfg_dict['IS_TRAINING']
        else:
            self.is_training = True

        # random seed, to make all experiments equal
        if random_seed is not None: # override random seed
            self.random_seed = random_seed
        elif 'RANDOM_SEED' in cfg_dict:
            self.random_seed = cfg_dict['RANDOM_SEED']
        else:
            self.random_seed = None

        # main task to perform, if unfilled, used regular
        if 'MODE' in cfg_dict:
            self.mode = cfg_dict['MODE']
        else:
            self.mode = 'regular'

        # main task to perform, if unfilled, used regular
        if 'OUTPUT_NAME' in cfg_dict:
            self.output_name = cfg_dict['OUTPUT_NAME']
        else:
            self.output_name = ""

        # add sub-config structures
        self.data = ConfigDataManager(cfg_dict=cfg_dict)
        self.model = ConfigModelManager(cfg_dict=cfg_dict)
        self.training = ConfigTrainingManager(cfg_dict=cfg_dict)
        self.system = ConfigSystemManager(cfg_dict=cfg_dict)

class ConfigDataManager(object):

    def __init__(self, cfg_dict):
        super(ConfigDataManager, self).__init__()

        # create sub-config structure
        self.path = cfg_dict['DATA']['PATH']
        self.train_files = cfg_dict['DATA']['TRAIN_FILES']
        self.test_files = cfg_dict['DATA']['TEST_FILES']

        # set number of inputs to append
        if 'APPEND' in cfg_dict['DATA']:
            self.append = cfg_dict['DATA']['APPEND']
        else:
            self.append = 1

        # set step between two inputs
        if 'STEP' in cfg_dict['DATA']:
            self.step = cfg_dict['DATA']['STEP']
        else:
            self.step = 1

        # set number of inputs to remove between two inputs
        if 'STRIDE' in cfg_dict['DATA']:
            self.stride = cfg_dict['DATA']['STRIDE']
        else:
            self.stride = 1

        # if to share part of the data from test with train. choose 0.0 for no sharing
        if 'SHARE_TRAIN' in cfg_dict['DATA']:
            self.share_train = cfg_dict['DATA']['SHARE_TRAIN']
        else:
            self.share_train = 0.0

        # if to share a different fold of the data, rather than the last one
        if 'KFOLD' in cfg_dict['DATA']:
            self.kfold = cfg_dict['DATA']['KFOLD']
        else:
            self.kfold = None

        # decide if single test it is or not
        if 'SINGLE_TEST' in cfg_dict['DATA']:
            self.single_test = cfg_dict['DATA']['SINGLE_TEST']
        else:
            self.single_test = False

        # decide if to shuffle data or not
        if 'SHUFFLE' in cfg_dict['DATA']:
            self.shuffle = cfg_dict['DATA']['SHUFFLE']
        elif self.single_test:
            self.shuffle = False
        else:
            self.shuffle = True

        # main task to perform, if unfilled, used regular
        if 'LABEL_PERCENTAGE' in cfg_dict['DATA']:
            self.label_percentage = cfg_dict['DATA']['LABEL_PERCENTAGE']
        else:
            self.label_percentage = 0.7

        # if to leave a subject out
        if 'LEAVE_SUBJECT_OUT' in cfg_dict['DATA']:
            self.leave_subject_out = cfg_dict['DATA']['LEAVE_SUBJECT_OUT']
        else:
            self.leave_subject_out = None

        # if to constrain the number of subjects for training
        if 'FORCE_NUM_SUBJECTS_TRAIN' in cfg_dict['DATA']:
            self.force_num_subjects_train = cfg_dict['DATA']['FORCE_NUM_SUBJECTS_TRAIN']
        else:
            self.force_num_subjects_train = None

        # compute DTW features for train and test datasets
        if 'DTW' in cfg_dict['DATA']:
            self.dtw = cfg_dict['DATA']['DTW']
        else:
            self.dtw = False

        # filter settings for noise reduction
        if 'APPLY_FILTER' in cfg_dict['DATA']:
            self.apply_filter = cfg_dict['DATA']['APPLY_FILTER']
        else:
            self.apply_filter = False

        if 'FILTER_CUTOFF' in cfg_dict['DATA']:
            self.filter_cutoff = cfg_dict['DATA']['FILTER_CUTOFF']
        else:
            self.filter_cutoff = 15  # Default cutoff frequency in Hz

        if 'FILTER_ORDER' in cfg_dict['DATA']:
            self.filter_order = cfg_dict['DATA']['FILTER_ORDER']
        else:
            self.filter_order = 4  # Default filter order

        # device related settings
        if 'CLASSES' in cfg_dict['DATA']:
            self.classes = cfg_dict['DATA']['CLASSES']
        else:
            self.classes = None

class ConfigModelManager(object):

    def __init__(self, cfg_dict):
        super(ConfigModelManager, self).__init__()

        # define model type
        if 'TYPE' in cfg_dict['MODEL']:
            self.type = cfg_dict['MODEL']['TYPE']
        else:
            self.type = 'neuralnet'

        # if to add residual layers or not
        if 'CLASSIFIER' in cfg_dict['MODEL']:
            self.classifier = cfg_dict['MODEL']['CLASSIFIER']
        else:
            self.classifier = None

        # add number of fc layers
        if 'NUM_FC_LAYERS' in cfg_dict['MODEL']:
            self.num_fc_layers = cfg_dict['MODEL']['NUM_FC_LAYERS']
        else:
            self.num_fc_layers = 2

        # if to add residual layers or not
        if 'WEIGHTS' in cfg_dict['MODEL']:
            self.weights = cfg_dict['MODEL']['WEIGHTS']
        else:
            self.weights = "" # default

        # if to add residual layers or not
        if 'EMBEDDING_WEIGHTS' in cfg_dict['MODEL']:
            self.embedding_weights = cfg_dict['MODEL']['EMBEDDING_WEIGHTS']
        else:
            self.embedding_weights = "" # default

        # if to send intermediate layer output during inference
        if 'SEND_INTERMEDIATE' in cfg_dict['MODEL']:
            self.send_intermediate = cfg_dict['MODEL']['SEND_INTERMEDIATE']
        else:
            self.send_intermediate = False

class ConfigTrainingManager(object):

    def __init__(self, cfg_dict):
        super(ConfigTrainingManager, self).__init__()

        # create sub-config structure
        # number of samples in one batch
        if 'BATCH_SIZE' in cfg_dict['TRAINING']:
            self.batch_size = cfg_dict['TRAINING']['BATCH_SIZE']
        else:
            self.batch_size = 32

        # number of epochs for training
        if 'EPOCHS' in cfg_dict['TRAINING']:
            self.epochs = cfg_dict['TRAINING']['EPOCHS']
        else:
            self.epochs = 40

        # number of epochs between each checkpoint saving
        if 'CP_INTERVAL' in cfg_dict['TRAINING']:
            self.cp_interval = cfg_dict['TRAINING']['CP_INTERVAL']
        else:
            self.cp_interval = 20

        # learning rate for optimizer
        if 'LEARNING_RATE' in cfg_dict['TRAINING']:
            self.learning_rate = cfg_dict['TRAINING']['LEARNING_RATE']
        else:
            self.learning_rate = 1e-3

        # weight decay for optimizer
        if 'WEIGHT_DECAY' in cfg_dict['TRAINING']:
            self.weight_decay = cfg_dict['TRAINING']['WEIGHT_DECAY']
        else:
            self.weight_decay = None

        # weight decay for optimizer
        if 'EPSILON' in cfg_dict['TRAINING']:
            self.epsilon = cfg_dict['TRAINING']['EPSILON']
        else:
            self.epsilon = None

        # weight decay for optimizer
        if 'MOMENTUM' in cfg_dict['TRAINING']:
            self.momentum = cfg_dict['TRAINING']['MOMENTUM']
        else:
            self.momentum = None

        # weight decay for optimizer
        if 'SCHEDULER_STEPS' in cfg_dict['TRAINING']:
            self.scheduler_steps = cfg_dict['TRAINING']['SCHEDULER_STEPS']
        else:
            self.scheduler_steps = None

        # weight decay for optimizer
        if 'GRADIENT_CLIP' in cfg_dict['TRAINING']:
            self.gradient_clip = cfg_dict['TRAINING']['GRADIENT_CLIP']
        else:
            self.gradient_clip = None

        # set num of workers for batch loading
        if 'BATCH_NUM_WORKERS' in cfg_dict['TRAINING']:
            self.batch_num_workers = cfg_dict['TRAINING']['BATCH_NUM_WORKERS']
        else:
            self.batch_num_workers = 1

        # weight decay for optimizer
        if 'WEIGHTED_SAMPLING' in cfg_dict['TRAINING']:
            self.weighted_sampling = cfg_dict['TRAINING']['WEIGHTED_SAMPLING']
        else:
            self.weighted_sampling = False

class ConfigSystemManager(object):

    def __init__(self, cfg_dict, gpu=None):
        super(ConfigSystemManager, self).__init__()

        # create sub-config structure
        # gpu index to execute session on
        if gpu is not None:
            self.gpu = gpu
        elif 'GPU' in cfg_dict['SYSTEM']:
            self.gpu = cfg_dict['SYSTEM']['GPU']
        else:
            self.gpu = 0

        # memory requirement. if not used, 0 will not check for available memory in gpu
        if 'MEMORY_REQ' in cfg_dict['SYSTEM']:
            self.memory_req = cfg_dict['SYSTEM']['MEMORY_REQ']
        else:
            self.memory_req = 0

# set random seed in all platforms
def set_random_seed(cfg):
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
        np.random.seed(seed=cfg.random_seed)


# create model based on config choice
def initiate_model(cfg):

    if cfg.model.classifier == 'lda':
        return LinearDiscriminantAnalysis()
    elif cfg.model.classifier == 'rf':
        return RandomForestClassifier(n_estimators=8, max_depth=8)
    elif cfg.model.classifier == 'dt':
        return DecisionTreeClassifier(max_depth=10)
    elif cfg.model.classifier == 'mlp':
        return MLPClassifier()
    elif cfg.model.classifier == 'svc':
        return LinearSVC()
    elif cfg.model.classifier == 'nb':
        return GaussianNB()
    elif cfg.model.classifier == 'knn':
        return KNeighborsClassifier()
    else:
        raise ValueError('Model was not found!')

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, device='cpu', bandwidth=None):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):

    def __init__(self, device='cpu', kernel=None):
        super().__init__()
        self.device = device
        self.kernel = RBF(device=device)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY