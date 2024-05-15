import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import hyperopt
from hyperopt import fmin, hp, Trials
from manual_submission import NN2, NN2_Stage2
import torch.cuda

MEAN = np.array([6.98823529e-01, 3.10588235e-01, 5.69411765e-01, 3.12756376e+00,
                 8.34544177e+01, 4.43847572e+02, 6.42282353e+01])

STD = np.array([0.45876923, 0.46273446, 0.49515857, 0.35491092, 19.91569537,
                75.2047422, 25.23083367])
MEAN = MEAN.reshape(1, -1)
STD = STD.reshape(1, -1)

MEAN_2 = np.array([6.87500000e-01, 1.77083333e-01, 4.53125000e-01, 3.07512500e+00,
                   8.66493195e+01, 4.34206375e+02, 6.61038368e+01, 6.25364583e+01])

STD_2 = np.array([0.46351241, 0.38173921, 0.49779789, 0.3413766, 4.69034568,
                  68.83339538, 19.16783255, 26.62213842])
MEAN_2 = MEAN_2.reshape(1, -1)
STD_2 = STD_2.reshape(1, -1)
MUST_TO_HAVE_PHASE1 = ["year", "is_fall", "is_boun", "is_ie", "gpa", "ales", "uni_score"]
MUST_TO_HAVE_PHASE2 = MUST_TO_HAVE_PHASE1 + ["interview_score"]


def check_df_requirements(df, stage):
    required_columns = MUST_TO_HAVE_PHASE1 if stage == 1 else MUST_TO_HAVE_PHASE2

    missing_columns = [col for col in required_columns if col not in df.columns]
    non_numeric_columns = [col for col in required_columns if
                           col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    if missing_columns:
        return (False, missing_columns)
    elif non_numeric_columns:
        return (False, non_numeric_columns)
    else:
        return (True, "")


def stage1_preds(df):
    with_test = False
    with_test2 = False

    if "result_1" in df.columns:
        with_test = True
    if "result_2" in df.columns:
        with_test2 = True
        y_result2 = df["result_2"].values
        df = df.drop(columns=['result_2'])
    if "interview_score" in df.columns:
        df = df.drop(columns=['interview_score'])
    df['n_of_applicants'] = df.groupby(['year', 'is_fall'])['year'].transform('count')
    df.drop(columns=['year'], inplace=True)


    X_orig = df
    if with_test:
       X_orig = df.drop(columns=['result_1'])
       y = df['result_1'].values

    # Split the data into train and test sets with random_state
    scaler = StandardScaler()
    scaler.mean_ = MEAN
    scaler.scale_ = STD
    X = scaler.transform(X_orig.values)
    X_tensor = torch.from_numpy(X).float()
    if with_test:
       y_tensor = torch.from_numpy(y).float()
    # Initialize model and load state dict
    model = NN2(0.603)

    model.load_state_dict(torch.load("NN_model.pt", map_location=torch.device('cpu')))
    st.success(f"NN Model loaded successfully...")

    # Get predictions
    with torch.no_grad():
        test_outputs = model(X_tensor)

        _, predicted = torch.max(test_outputs, 1)
        if with_test:
            test_accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            # Apply softmax to get probabilities
        test_probs = torch.softmax(test_outputs, dim=1)
        st.success(f"Predictions ready!")
        test_probs_np = test_probs.cpu().numpy()
        # Create new columns for each class probability in X_test
        label_map = {0: "Accept", 1: "Reject", 2: "Interview"}
        label_map2 =  {0: "Rejected After Interview", 1: "Accepted After Interview"}
        X_orig["pred"] = [label_map[int(i)] for i in np.array(predicted)]
        for i in range(test_probs_np.shape[1]):
            X_orig[f"prob_{label_map[i]}"] = test_probs_np[:, i]
        if with_test:

            if with_test2:
                X_orig["actual"] = [label_map[int(i)] for i in y]
                y_r2 = []
                for i in y_result2:
                    try:
                        y_r2.append(label_map2[int(i)])
                    except:
                        y_r2.append(None)
                X_orig["result_after_interview"] = y_r2
        st.success(f"Excel file ready!")


    return X_orig

def stage2_preds(df):
    with_test = False
    df = df.dropna()
    if "result_1" in df.columns:
        df.drop(columns=['result_1'], inplace=True)
    if "result_2" in df.columns:
        with_test = True
    df['n_of_applicants'] = df.groupby(['year', 'is_fall'])['year'].transform('count')
    df.drop(columns=['year'], inplace=True)

    X_orig = df
    if with_test:
       X_orig = df.drop(columns=['result_2'])
       y = df['result_2'].values

    # Split the data into train and test sets with random_state
    scaler = StandardScaler()
    scaler.mean_ = MEAN_2
    scaler.scale_ = STD_2
    X = scaler.transform(X_orig.values)
    X_tensor = torch.from_numpy(X).float()
    # Initialize model and load state dict
    model = NN2_Stage2(0.603)

    model.load_state_dict(torch.load("NN_stage2.pt", map_location=torch.device('cpu')))
    st.success(f"NN Model loaded successfully...")

    # Get predictions
    with torch.no_grad():
        test_outputs = model(X_tensor)
        test_predictions = (test_outputs >= 0.5).float()
        if with_test:
            X_orig["actual"] = y
            X_orig["actual"] = X_orig["actual"].astype(int)
        X_orig["pred"] = np.array(test_predictions)
        X_orig["probabilities"] = np.array(test_outputs)
        st.success(f"Excel file ready!")

    X_orig["pred"] = X_orig["pred"].astype(int)
    return X_orig
