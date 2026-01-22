from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Data preprocessing utilities
# ----------------------------
def split_segments(inputs: np.ndarray, targets: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split long EEG segments into non-overlapping windows of fixed length.

    Expected input shape:
        (num_segments, total_timepoints, num_channels, extra_dim)

    Returns:
        (Nwindows, window_size, num_channels) for both inputs and targets.
    """
    num_segments, total_timepoints, num_channels, extra_dim = inputs.shape
    num_splits = total_timepoints // window_size

    split_inputs = inputs.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)
    split_targets = targets.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)

    split_inputs = split_inputs.squeeze(-1)
    split_targets = split_targets.squeeze(-1)

    split_inputs = split_inputs.reshape(-1, window_size, num_channels)
    split_targets = split_targets.reshape(-1, window_size, num_channels)

    return split_inputs, split_targets


def select_channels_per_patient(
    x_patient: np.ndarray,
    y_patient: np.ndarray,
    patient_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2-channel bipolar montage per patient (left or right hemisphere).

    Left hemisphere uses channels:
      F7(10), T7(12), P7(14)
    Right hemisphere uses channels:
      F8(11), T8(13), P8(15)

    Output channels:
      1) F - T
      2) P - T

    Returns:
      x_selected, y_selected with shape (Nwindows, S, 2).
    """
    channel_dict = {
        "data1": 1,
        "data2": 0
    }

    if patient_id not in channel_dict:
        raise ValueError(f"Patient ID {patient_id} not found in channel_dict.")

    side_flag = channel_dict[patient_id]

    if side_flag == 0:
        fx, tx, px = x_patient[:, :, 10], x_patient[:, :, 12], x_patient[:, :, 14]
        fy, ty, py = y_patient[:, :, 10], y_patient[:, :, 12], y_patient[:, :, 14]
    elif side_flag == 1:
        fx, tx, px = x_patient[:, :, 11], x_patient[:, :, 13], x_patient[:, :, 15]
        fy, ty, py = y_patient[:, :, 11], y_patient[:, :, 13], y_patient[:, :, 15]
    else:
        raise ValueError(f"Invalid side_flag '{side_flag}' for patient {patient_id}")

    x_selected = np.stack([fx - tx, px - tx], axis=-1)
    y_selected = np.stack([fy - ty, py - ty], axis=-1)

    return x_selected, y_selected


# ----------------------------
# Data loading
# ----------------------------
def load_all_patients(data_base_dir: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all patients' paired input/target segments, window them, and apply montage selection.

    Returns:
      all_inputs:  (N, window_size, 2)
      all_targets: (N, window_size, 2)
      patient_ids: list[str] length N
    """
    patient_folders = sorted(glob.glob(os.path.join(data_base_dir, "Filtered_Data_pat*")))
    if not patient_folders:
        raise FileNotFoundError("No patient folders found in Data directory.")

    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    patient_ids: List[str] = []

    for patient_folder in patient_folders:
        patient_id_folder = os.path.basename(patient_folder)

        input_files = sorted(glob.glob(os.path.join(patient_folder, "original_filtered_segment_*.npy")))
        target_files = sorted(
            glob.glob(os.path.join(data_base_dir, patient_id_folder.replace("Filtered_", ""), "preprocessed_segment_*.npy"))
        )

        if not input_files or not target_files:
            continue

        inputs = [np.load(f) for f in input_files]
        targets = [np.load(f) for f in target_files]

        patient_id = patient_id_folder.split("Filtered_Data_")[-1]

        x_patient, y_patient = split_segments(np.array(inputs), np.array(targets), window_size)
        x_patient, y_patient = select_channels_per_patient(x_patient, y_patient, patient_id)

        all_inputs.append(x_patient)
        all_targets.append(y_patient)
        patient_ids.extend([patient_id] * len(x_patient))

    all_inputs_arr = np.concatenate(all_inputs, axis=0)
    all_targets_arr = np.concatenate(all_targets, axis=0)

    return all_inputs_arr, all_targets_arr, patient_ids


# ----------------------------
# Loss
# ----------------------------
class RRMSELoss(nn.Module):
    """
    Relative RMSE loss:
        RMSE(y_pred, y_true) / RMS(y_true)

    A small epsilon is added to RMS(y_true) for numerical stability.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
        rms_y = torch.sqrt(torch.mean(y_true**2)) + 1e-8
        return rmse / rms_y


# ----------------------------
# Training
# ----------------------------
def train_model_full(
    model: torch.nn.Module,
    model_id: str,
    train_loader,
    loss_function,
    device: torch.device,
    num_epochs: int,
    lr: float,
    model_save_path: str,
) -> torch.nn.Module:
    """
    Train on ALL data (no validation) and save final weights to:
        {model_save_path}/{model_id}/best_model.pth
    """
    os.makedirs(model_save_path, exist_ok=True)

    model_folder = os.path.join(model_save_path, f"{model_id}")
    os.makedirs(model_folder, exist_ok=True)
    save_path_model = os.path.join(model_folder, "best_model.pth")

    optimizer = optim.Adam(model.parameters(), lr)
    criterion = loss_function

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch, _idx_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path_model)
    print(f"[INFO] Final model saved to: {save_path_model}")

    return model
