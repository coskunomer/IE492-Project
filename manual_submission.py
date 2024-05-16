import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

MEAN = np.array([6.98823529e-01, 3.10588235e-01, 5.69411765e-01, 3.12756376e+00,
       8.34544177e+01, 4.43847572e+02, 6.42282353e+01])

STD = np.array([ 0.45876923,  0.46273446,  0.49515857,  0.35491092, 19.91569537,
       75.2047422 , 25.23083367])

MEAN_2 = np.array([6.87500000e-01, 1.77083333e-01, 4.53125000e-01, 3.07512500e+00,
       8.66493195e+01, 4.34206375e+02, 6.61038368e+01, 6.25364583e+01])

STD_2 = np.array([ 0.46351241,  0.38173921,  0.49779789,  0.3413766 ,  4.69034568,
       68.83339538, 19.16783255, 26.62213842])



class NN2(nn.Module):
    def __init__(self, dropout):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer directly from 64 to 3
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 2 hidden layer NN
class NN2_Stage2(nn.Module):
    def __init__(self, dropout):
        super(NN2_Stage2, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def get_preds(is_boun, is_ie, term, gpa, ales, uni_score, n_of_applicants, interview_score=None):
    if (interview_score is None):
        if ales < 75:
            return np.array([0, 1, 0])
        # Create a StandardScaler object and set mean and standard deviation
        scaler = StandardScaler()
        scaler.mean_ = MEAN
        scaler.scale_ = STD
        term = 1 if term == "Fall" else 0
        features = np.array([term, is_boun, is_ie, gpa, ales, uni_score, n_of_applicants], dtype=np.float32)
        features = scaler.transform(features.reshape(1, -1))  # Reshape to (1, 7) as expected by StandardScaler

        x = torch.tensor(features, dtype=torch.float32)

        # Initialize model and load state dict
        model = NN2(0.603)
        model.load_state_dict(torch.load("NN1_new.pt", map_location=torch.device('cpu')))
        model.eval()

        # Get predictions
        with torch.no_grad():
            preds = model(x)
            preds = torch.softmax(preds, dim=1)
        return preds.numpy()[0]
    else:
        # Create a StandardScaler object and set mean and standard deviation
        scaler = StandardScaler()
        scaler.mean_ = MEAN_2
        scaler.scale_ = STD_2
        term = 1 if term == "Fall" else 0
        features = np.array([term, is_boun, is_ie, gpa, ales, uni_score, interview_score, n_of_applicants], dtype=np.float32)
        features = scaler.transform(features.reshape(1, -1))
        x = torch.tensor(features, dtype=torch.float32)
        model = NN2_Stage2(0.603)
        model.load_state_dict(torch.load("NN2_new.pt", map_location=torch.device('cpu')))
        model.eval()
        # Get predictions
        with torch.no_grad():
            preds = model(x)
            preds = np.array(preds)[0][0]
        return np.array([preds, 1-preds])




