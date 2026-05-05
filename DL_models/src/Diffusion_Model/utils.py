import torch
import numpy as np



def compute_mae(mean_sample, target, vitals):
    """
    samples: [B, T, D] (numpy, ORIGINAL SCALE)
    target:  [B, T, D]
    """

    maes = {}

    for i, vital in enumerate(vitals):

        pred_i = mean_sample[..., i]
        target_i = target[..., i]

        # MAE
        mae = np.mean(np.abs(pred_i - target_i), axis=1) #np.mean(np.abs(pred_i - target_i))

        maes[vital] = mae

    return maes


def compute_crps(samples, target, vitals):
    """
    samples: [S, B, T, D]
    target:  [B, T, D]
    """

    crpss = {}

    S = samples.shape[0]

    for i, vital in enumerate(vitals):

        samples_i = samples[..., i]  # [S, B, T]
        target_i = target[..., i]   # [B, T]

        # term1: E|X - y|
        term1 = np.mean(np.abs(samples_i - target_i[None, ...]), axis=0)

        # term2: E|X - X'|
        s1 = samples_i[:, None, :, :]  # [S,1,B,T]
        s2 = samples_i[None, :, :, :]  # [1,S,B,T]

        term2 = np.mean(np.abs(s1 - s2), axis=(0,1))

        crps = term1 - 0.5 * term2

        crpss[vital] = np.mean(crps, axis=1)#np.mean(crps)

    return crpss
