import os
import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    Parse ASVspoof protocol file and return file list with labels
    
    Args:
        dir_meta: Path to protocol file
        is_train: Whether this is training data
        is_eval: Whether this is evaluation data
    
    Returns:
        For training/dev: (labels_dict, file_list)
        For evaluation: file_list only
    """
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    
    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list




def pad(x, max_len=64600):
    """
    Pad audio to fixed length by repeating if necessary
    
    Args:
        x: Audio array
        max_len: Target length (default: 64600 samples = ~4 seconds at 16kHz)
    
    Returns:
        Padded audio array
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Repeat audio to fill length
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
			

class Dataset_ASVspoof2019_train(Dataset):
    """
    ASVspoof 2019 training dataset with RawBoost augmentation
    """
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        """
        Args:
            args: Arguments containing RawBoost parameters
            list_IDs: List of utterance IDs
            labels: Dictionary mapping utterance ID to label (0=spoof, 1=bonafide)
            base_dir: Base directory containing audio files
            algo: RawBoost algorithm (0-8)
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600  # ~4 sec audio at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + utt_id + '.flac', sr=16000)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target
            
            
class Dataset_ASVspoof2021_eval(Dataset):
    """
    ASVspoof 2021 evaluation dataset (no augmentation)
    """
    def __init__(self, list_IDs, base_dir):
        """
        Args:
            list_IDs: List of utterance IDs
            base_dir: Base directory containing audio files
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # ~4 sec audio at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + utt_id + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_in_the_wild_eval(Dataset):
    """
    In-the-Wild evaluation dataset (no augmentation)
    """
    def __init__(self, list_IDs, base_dir):
        """
        Args:
            list_IDs: List of audio file paths
            base_dir: Base directory containing audio files
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # ~4 sec audio at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id

def process_Rawboost_feature(feature, sr, args, algo):
    """
    Apply RawBoost data augmentation to audio
    
    Args:
        feature: Audio waveform
        sr: Sample rate
        args: Arguments containing augmentation parameters
        algo: Algorithm selection (0-8)
            0: No augmentation
            1: LnL_convolutive_noise
            2: ISD_additive_noise
            3: SSI_additive_noise (default)
            4: All three in series (1+2+3)
            5: 1+2 in series
            6: 1+3 in series
            7: 2+3 in series
            8: 1 and 2 in parallel
    
    Returns:
        Augmented audio waveform
    """
    # Convolutive noise
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
    
    # Impulsive noise
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    
    # Coloured additive noise
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # All 3 algorithms in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # First two algorithms in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    
    # First and third algorithms in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # Second and third algorithms in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # First two algorithms in parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                         args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                         args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # Normalize resultant waveform
    
    # No augmentation
    else:
        feature = feature
    
    return feature
