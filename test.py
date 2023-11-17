import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, file_name: str = "datasets/transactions.csv"):
        df = self.get_df(file_name)
        print(df["client_id"].nunique())
        df = df.groupby("client_id")

        dataset = []

        for group in df.groups:
            d: pd.DataFrame = df.get_group(group)

            d = d.drop(labels=["client_id", 'trans_time', 'trans_city', 'term_id'], axis=1)
            dataset.append(d.values.tolist())

        self.df = dataset


        # dataset = np.array(dataset, dtype="object")
        # print(dataset.shape)

        # train = torch.LongTensor(dataset)
        # print(train.shape())
        # self.df = self.preprocess_df(self.get_df(file_name))
        # self.X, self.y = self.XY_split()

    @staticmethod
    def get_df(file_name: str = "datasets/transactions.csv") -> pd.DataFrame:
        return pd.read_csv(file_name)

    @staticmethod
    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=["RowNumber", "CustomerId", "Surname"], axis=1)

        ohe = OneHotEncoder()
        tf = ohe.fit_transform(df[["Geography"]])
        df[ohe.categories_[0]] = tf.toarray()

        ohe = OneHotEncoder(drop=["Female"])
        tf = ohe.fit_transform(df[["Gender"]])
        df["Male"] = tf.toarray()

        return df.drop(labels=["Geography", "Gender"], axis=1)

    def XY_split(self):
        return self.df.drop(labels=["Exited"], axis=1), self.df["Exited"]


class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, sequence_length, hidden_size)

        # Taking the output from the last time step
        last_output = gru_out[:, -1]

        output = self.fc(last_output)
        # output shape: (batch_size, output_size)

        return F.sigmoid(output)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


# Example usage:
input_size = 3
hidden_size = 10
output_size = 1

model = GRUNetwork(input_size, hidden_size, output_size)

# Assuming you have input data x with shape (batch_size, sequence_length, input_size)
dp = DataProcessor()
print(dp.df[0])

print()
x = torch.Tensor(dp.df[0])  # Example input data

output = model(x)
print(output.shape)
