from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pyedflib
import torch
from scipy.signal import resample

import seResNet


def extract_number(file_path: str) -> int:
    """
    Extract the last integer found in a filename.

    This is useful for sorting files such as:
        "record_1.edf", "record_2.edf", ..., "record_10.edf"

    Parameters
    ----------
    file_path : str
        Path to a file.

    Returns
    -------
    int
        The last group of digits found in the basename. Returns 0 if none exist.
    """
    filename = os.path.basename(file_path)
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else 0


def extract_segment_by_time(
    file_path: str,
    start_time,
    end_time,
    target_freq: float = 256,
    window_size_sec: float = 5,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Extract EEG between start_time and end_time, resample, and split into windows.

    Parameters
    ----------
    file_path : str
        Path to EDF file.
    start_time : datetime.datetime
        Absolute time (same timezone as EDF start) where the segment begins.
    end_time : datetime.datetime
        Absolute time where the segment ends.
    target_freq : float, default 256
        Target sampling frequency after resampling (Hz).
    window_size_sec : float, default 5
        Length of each non-overlapping window in seconds.

    Returns
    -------
    segments : np.ndarray or None
        Windowed data of shape (N, S, C):
        - N: number of windows
        - S: samples per window
        - C: number of channels
    feature_datetimes : np.ndarray or None
        Shape (N,): datetime of the center of each window.
    segment_raw : np.ndarray or None
        Continuous raw segment at original sampling frequency, shape (C, T_raw).
    segment_resampled : np.ndarray or None
        Continuous resampled segment at target_freq, shape (C, T_resampled).

    Notes
    -----
    Returns (None, None, None, None) if:
    - requested time bounds are invalid/out of file bounds
    - segment is too short to form at least one full window
    - any read/resample error occurs
    """
    f = pyedflib.EdfReader(file_path)
    try:
        # Basic EDF info
        start_ts = f.getStartdatetime()
        duration_sec = f.file_duration
        n_channels = f.signals_in_file
        n_samples = f.getNSamples()[0]
        sampling_freq = n_samples / duration_sec

        # Convert requested times to seconds from EDF start
        seg_start_sec = (start_time - start_ts).total_seconds()
        seg_end_sec = (end_time - start_ts).total_seconds()

        # Sanity checks
        if seg_start_sec < 0 or seg_end_sec > duration_sec or seg_start_sec >= seg_end_sec:
            print(f"[WARN] Requested segment out of bounds or invalid for: {file_path}")
            return None, None, None, None

        # Read full file (simple and robust)
        sigbufs = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            sigbufs[ch, :] = f.readSignal(ch)

        # Convert segment bounds to sample indices
        win_start = int(seg_start_sec * sampling_freq)
        win_end = int(seg_end_sec * sampling_freq)

        if win_start < 0 or win_end > sigbufs.shape[1] or win_start >= win_end:
            print(f"[WARN] Invalid window indices for: {file_path}")
            return None, None, None, None

        # Continuous raw segment (C, T_raw)
        segment_raw = sigbufs[:, win_start:win_end]

        # Resample continuous segment to target frequency
        n_samples_target = int(segment_raw.shape[1] * (target_freq / sampling_freq))
        segment_resampled = resample(segment_raw, n_samples_target, axis=1)  # (C, T_resampled)

        # Windowing parameters
        window_size_samples = int(window_size_sec * target_freq)
        total_samples = segment_resampled.shape[1]

        n_windows = total_samples // window_size_samples
        if n_windows == 0:
            print(f"[WARN] Segment too short for even one window in {file_path}")
            return None, None, None, None

        # Keep only full windows in resampled data
        n_samples_keep = n_windows * window_size_samples
        segment_resampled = segment_resampled[:, :n_samples_keep]  # (C, n_windows*S)

        # (C, T) -> (T, C) -> (N, S, C)
        segment_t_c = segment_resampled.T  # (T_keep, C)
        segments = segment_t_c.reshape(n_windows, window_size_samples, n_channels)

        # Datetimes for window centers
        segment_start_time = start_ts + timedelta(seconds=float(seg_start_sec))
        middle_offsets = np.arange(n_windows) * window_size_sec + (window_size_sec / 2.0)
        feature_datetimes = np.array(
            [segment_start_time + timedelta(seconds=float(s)) for s in middle_offsets]
        )

        print(
            f"Final segment shape: {segments.shape} (N,S,C), "
            f"raw segment: {segment_raw.shape}, resampled: {segment_resampled.shape}"
        )

        return segments, feature_datetimes, segment_raw, segment_resampled

    except Exception as e:
        print(f"[ERROR] Failed to process file {file_path}: {e}")
        return None, None, None, None
    finally:
        f.close()


def clean_segment(segment: np.ndarray, model: torch.nn.Module, sampling_freq: float) -> np.ndarray:
    """
    Clean a continuous multichannel EEG segment using an artifact removal model.

    The model is applied in contiguous chunks sized to match what it expects (defaulted here
    to 5 seconds). Each chunk is padded to a length divisible by 16 if required.

    Parameters
    ----------
    segment : np.ndarray
        EEG array of shape (C, T): channels x timepoints.
    model : torch.nn.Module
        Trained artifact removal model.
        Expected input shape: (1, T, C) and output shape compatible with (1, T, C).
    sampling_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Cleaned EEG array of shape (C, T), same length as input.
    """
    model_len_sec = 5  # seconds per chunk the model expects
    expected_len = int(model_len_sec * sampling_freq)
    expected_len -= expected_len % 16  # enforce divisibility by 16

    cleaned_all = []
    seg_len = segment.shape[1]

    for start_idx in range(0, seg_len, expected_len):
        end_idx = min(start_idx + expected_len, seg_len)
        seg_chunk = segment[:, start_idx:end_idx]  # (C, t)

        # Pad to multiple of 16 along time axis if needed
        pad_len = (16 - (seg_chunk.shape[1] % 16)) % 16
        if pad_len > 0:
            seg_chunk = np.pad(seg_chunk, ((0, 0), (0, pad_len)), mode="constant")

        # Model expects (1, time, channels)
        input_tensor = torch.tensor(seg_chunk.T, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            cleaned_chunk = model(input_tensor).squeeze(0).T.numpy()  # back to (C, time)

        if pad_len > 0:
            cleaned_chunk = cleaned_chunk[:, :-pad_len]

        cleaned_all.append(cleaned_chunk)

    return np.concatenate(cleaned_all, axis=1) if cleaned_all else segment


def clean_segments_array(
    segments: Optional[np.ndarray],
    model: torch.nn.Module,
    sampling_freq: float = 256,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply the artifact removal model to each window and also return a continuous reconstruction.

    Parameters
    ----------
    segments : np.ndarray or None
        Windowed EEG of shape (N, S, C), typically from `extract_segment_by_time`.
    model : torch.nn.Module
        Artifact removal model.
    sampling_freq : float, default 256
        Sampling frequency of the windowed signals in Hz.

    Returns
    -------
    cleaned_segments : np.ndarray or None
        Cleaned windowed EEG of shape (N, S, C).
    cleaned_continuous : np.ndarray or None
        Reconstructed continuous cleaned EEG of shape (C, N*S).

    Notes
    -----
    If `segments` is None, returns (None, None).
    """
    if segments is None:
        return None, None

    n_windows, n_samples, n_channels = segments.shape
    cleaned_list = []

    for i in range(n_windows):
        # (S, C) -> (C, S)
        seg_c_s = segments[i].T
        cleaned_c_s = clean_segment(seg_c_s, model, sampling_freq)
        cleaned_list.append(cleaned_c_s.T)  # back to (S, C)

    cleaned_segments = np.stack(cleaned_list, axis=0)  # (N, S, C)

    # (N, S, C) -> (C, N*S)
    cleaned_continuous = cleaned_segments.transpose(0, 2, 1).reshape(n_channels, n_windows * n_samples)

    return cleaned_segments, cleaned_continuous


def preprocess_data(edf_path: str, model_path: str, start, end) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing for a time-bounded EDF segment:
    1) Load artifact removal model weights
    2) Extract raw segment between `start` and `end`
    3) Resample + window into 5s chunks (via `extract_segment_by_time`)
    4) Clean each chunk and reconstruct a continuous cleaned signal

    Parameters
    ----------
    edf_path : str
        Path to the EDF file to read from.
    model_path : str
        Path to the saved PyTorch model weights (state_dict).
    start : datetime.datetime
        Segment start datetime.
    end : datetime.datetime
        Segment end datetime.

    Returns
    -------
    raw_data : np.ndarray
        Continuous resampled (but uncleaned) data of shape (C, T).
    preprocessed_data : np.ndarray
        Continuous cleaned data of shape (C, T).

    Raises
    ------
    ValueError
        If extraction fails (e.g., out-of-bounds request).
    """
    model = seResNet.SE_ResNet1D(
        2, 5, [32, 64, 128, 256, 512], [9, 7, 5, 3, 3], 4, True, 0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    segments, _, _, raw_data = extract_segment_by_time(
        file_path=edf_path,
        start_time=start,
        end_time=end,
    )

    if segments is None or raw_data is None:
        raise ValueError("Segment extraction failed (segments/raw_data is None). Check time bounds and EDF file.")

    _, preprocessed_data = clean_segments_array(
        segments=segments,
        model=model,
        sampling_freq=256,
    )

    if preprocessed_data is None:
        raise ValueError("Cleaning failed (preprocessed_data is None).")

    return raw_data, preprocessed_data


