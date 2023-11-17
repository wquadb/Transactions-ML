import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm


class DataProcessor:
    def __init__(self, file_name: str = "datasets/transactions.csv"):
        df = self.get_df(file_name)
        print(df["client_id"].nunique())
        df = df.groupby("client_id")

        dataset = []

        for group in df.groups:
            d: pd.DataFrame = df.get_group(group)

            d = d.drop(labels=['trans_time', 'trans_city', 'term_id'], axis=1)
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
        last_output = gru_out[:, -1, :]

        output = self.fc(last_output)
        # output shape: (batch_size, output_size)

        return F.sigmoid(output)


def get_train():
    return pd.read_csv("datasets/train.csv")


def train_model(nn: GRUNetwork, criterion, optimizer, train, dp):
    for epoch in range(2):
        print(f"Epoch {epoch}")

        nn.train()
        loss = 0

        answers = 0
        allans = 0

        for X in tqdm(dp.df):
            clid = X[0][0]

            if not train[train["client_id"] == clid].empty:
                y = torch.Tensor([int(train[train["client_id"] == clid]["gender"])]).view(1, 1)
            else:
                continue

            for i in range(len(X)):
                X[i] = X[i][1:]
            X = torch.Tensor([X])

            # Forward
            y_pred = nn(X)

            # Loss computation
            loss = criterion(y_pred, y)

            # Backward
            loss.backward()

            # Update weights
            optimizer.step()

            # Zero gradients
            optimizer.zero_grad()

            # log
            acc = (y_pred.round() == y).float()  # right answer
            answers += acc
            allans += 1

        print(answers / allans)

        # Debug
        # self.nn.eval()
        #
        # self.train_acc_history.append(self.test_model())
        # self.test_acc_history.append(self.eval_model())
        # self.loss_history.append(loss.item())

        # Printing provisional results
        # if epoch % 10 == 9 or epoch == 0:
        #     print(f"Epoch {epoch + 1}: Loss {loss.item():.4f}: Accuracy {self.test_acc_history[-1]}")

    print("Finished Training")


train = get_train()


# Example usage:
input_size = 3
hidden_size = 10
output_size = 1

model = GRUNetwork(input_size, hidden_size, output_size)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dp = DataProcessor()

train_model(model, criterion, optimizer, train, dp=dp)
exit(0)

# Assuming you have input data x with shape (batch_size, sequence_length, input_size)
print(dp.df[0])

x = torch.Tensor([dp.df[1]])  # Example input data
print(x.shape, "shape")

output = model(x)
print(output.shape)
