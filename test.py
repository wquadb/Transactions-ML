from modules import process as pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import argparse
import random
import json
import os


device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print("\nStarting with device:", device)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) cuda seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}\n")


def preprocess_data():
    # region ----- functions -----
    def df_one_hot(df, column):
        one_hot = pd.get_dummies(df[column], dtype=float)
        df = df.drop(column, axis=1)
        df = df.join(one_hot)
        return df

    # normalizes df values from integer to float(0-1)
    def df_normalize(df, column):
        max_value = df[column].max()
        min_value = df[column].min()
        df[column] = (df[column] - min_value) / (max_value - min_value)
        return df

    # region ----- generate x table -----
    df_x = pd.read_csv("datasets/transactions.csv")

    # transform time
    # df_x[['days', 'other_date']] = df_x['trans_time'].str.split(' ', expand=True)
    # df_x[['hours', 'minutes', 'seconds']] = df_x['other_date'].str.split(':', expand=True)
    df_x.drop(["term_id"], axis=1, inplace=True)
    df_x = string_to_date(df=df_x, column='trans_time')

    # transform trans_city
    df_x = df_one_hot(df_x, 'trans_city')

    # transform amount
    df_x = df_x[df_x['amount'] < 4000000]
    df_x = df_normalize(df_x, 'amount')

    # transform mcc_code
    df_x['mcc_code'] = df_x['mcc_code'].apply(
        lambda x: f"mcc_code_{str(x)[:2]}" if str(x)[0] == '5' or str(x)[0] == '7' else f"mcc_code_{str(x)[0]}")
    df_x = df_one_hot(df_x, 'mcc_code')

    # transform trans_types
    df_x['trans_type'] = df_x['trans_type'].apply(lambda x: f"trans_type_{str(x)[0]}")
    df_x = df_one_hot(df_x, 'trans_type')

    return df_x


class DataProcessor:
    def __init__(self, file_name: str = "datasets/transactions.csv", valid: str = "datasets/train.csv", predict_from: int = 2500):
        self.valid = valid
        self.predict_from = predict_from
        df = self.get_df(file_name)
        df = preprocess_data()
        print(df.columns)

        self.X_train, self.y_train = self.get_train_dataset(df)

    def get_train(self):
        return pd.read_csv(self.valid)

    def get_train_dataset(self, df):
        print(f'Unique clients: {df["client_id"].nunique()}')
        df = df.groupby("client_id")

        X_train = []
        y_train = []

        train = self.get_train()

        for group in df.groups:
            d: pd.DataFrame = df.get_group(group)
            d = d[:self.predict_from]

            if not train[train["client_id"] == group].empty:
                y = train[train["client_id"] == group]["gender"].iloc[0]
                y_train.append(y)
            else:
                continue

            d = d.drop(labels=['client_id'], axis=1)
            X_train.append(d.values.tolist())

        X_train = pad_sequence([torch.Tensor(i[::-1]) for i in X_train], batch_first=True).flip(dims=[1]).to(device)
        y_train = torch.Tensor(y_train).view(-1, 1).to(device)

        print(f"X_Train_shape: {X_train.shape}")
        print(f"X_Train_shape: {y_train.shape}")

        return X_train, y_train

    @staticmethod
    def get_df(file_name: str = "datasets/transactions.csv") -> pd.DataFrame:
        return pd.read_csv(file_name)


class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.25),

            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        rnn_out, _ = self.gru(x)
        # gru_out shape: (batch_size, sequence_length, hidden_size)

        # Taking the output from the last time step
        last_output = rnn_out[:, -1, :]

        output = self.linear_layer(last_output)
        # output shape: (batch_size, output_size)

        return F.sigmoid(output)


class Worker:
    def __init__(self, settings: dict):
        self.settings = settings

        self.dataprocessor = DataProcessor(file_name=settings["transactions"], valid=settings["train"])

        self.dataset = TensorDataset(self.dataprocessor.X_train, self.dataprocessor.y_train)

        # Example usage:
        input_size = 42
        hidden_size = 28
        output_size = 1

        self.nn = GRUNetwork(input_size, hidden_size, output_size).to(device)

        self.train_acc_history = []
        self.test_acc_history = []
        self.loss_history = []

    def train_model(self):
        print("\nStarting Training")

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.settings["learning_rate"])

        loader = DataLoader(self.dataset, batch_size=self.settings["batch_size"], shuffle=True)

        # Training loop
        for epoch in range(self.settings["num_epochs"]):
            print(f"Epoch {epoch+1}")

            for X, y in tqdm(loader):
                # Forward
                y_pred = self.nn(X)

                # Loss computation
                loss = criterion(y_pred, y)

                # Backward
                loss.backward()

                # Update weights
                optimizer.step()

                # Zero gradients
                optimizer.zero_grad()

                # LOG
                acc = (y_pred.round() == y).float().mean().to("cpu")

                self.train_acc_history.append(acc)
                self.loss_history.append(loss.item())

    def show_stats(self):
        plt.plot(self.train_acc_history, label="Train")
        # plt.plot(self.test_acc_history, label="Test")

        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Batch")
        plt.grid()

        # if savefig:
        #     plt.savefig(savefig)
        # else:
        plt.show()

# def train_model(nn: GRUNetwork, criterion, optimizer, train, dp):
#     for epoch in range(2):
#         print(f"Epoch {epoch}")
#
#         nn.train()
#         loss = 0
#
#         answers = 0
#         allans = 0
#
#         for X in tqdm(dp.df):
#             clid = X[0][0]
#
#             if not train[train["client_id"] == clid].empty:
#                 y = torch.Tensor([int(train[train["client_id"] == clid]["gender"])]).view(1, 1)
#             else:
#                 continue
#
#             for i in range(len(X)):
#                 X[i] = X[i][1:]
#             X = torch.Tensor([X])
#
#             # Forward
#             y_pred = nn(X)
#
#             # Loss computation
#             loss = criterion(y_pred, y)
#
#             # Backward
#             loss.backward()
#
#             # Update weights
#             optimizer.step()
#
#             # Zero gradients
#             optimizer.zero_grad()
#
#             # log
#             acc = (y_pred.round() == y).float()  # right answer
#             answers += acc
#             allans += 1
#
#         print(answers / allans)
#
#         # Debug
#         # self.nn.eval()
#         #
#         # self.train_acc_history.append(self.test_model())
#         # self.test_acc_history.append(self.eval_model())
#         # self.loss_history.append(loss.item())
#
#         # Printing provisional results
#         # if epoch % 10 == 9 or epoch == 0:
#         #     print(f"Epoch {epoch + 1}: Loss {loss.item():.4f}: Accuracy {self.test_acc_history[-1]}")
#
#     print("Finished Training")


if __name__ == "__main__":
    set_seed()

    settings = {
        "transactions": "datasets/transactions.csv",
        "train": "datasets/train.csv",
        "batch_size": 378,
        "learning_rate": 0.001,
        "num_epochs": 50
    }

    worker = Worker(settings=settings)
    worker.train_model()
    worker.show_stats()

# Example usage:
# input_size = 3
# hidden_size = 10
# output_size = 1
#
# model = GRUNetwork(input_size, hidden_size, output_size)
#
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# dp = DataProcessor()
#
# train_model(model, criterion, optimizer, train, dp=dp)
# exit(0)
#
# # Assuming you have input data x with shape (batch_size, sequence_length, input_size)
# print(dp.df[0])
#
# x = torch.Tensor([dp.df[1]])  # Example input data
# print(x.shape, "shape")
#
# output = model(x)
# print(output.shape)
