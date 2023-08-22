import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from request_event import *

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"now using {device}...")


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        # print(hidden[0].shape, hidden[1].shape)  # torch.Size([1, 1, 256]) torch.Size([1, 1, 256])
        return hidden


def load_data():
    df = pd.read_csv("歷年台股大盤指數.csv", index_col=0)
    # print(df.info())
    df["Open"] = df["Open"].apply(lambda x: x.replace(",", "")).astype("float")
    # print(df["Open"].tail(60).values)

    return df["Open"].values


def predict(hidden_dim, epoch, num_layers):
    data = load_data()

    sc = MinMaxScaler(feature_range=(0, 1))
    data_scaled = sc.fit_transform(data.reshape(-1, 1))
    data_scaled = data_scaled.reshape(1, -1, 1)
    # print(data_scaled)
    # print(data_scaled.shape)  # (1, 6117, 1)

    last_sixty = data_scaled[:, -60:, :]
    # print(last_sixty)
    # print(last_sixty.shape)  # (1, 60, 1)

    hidden_dim = int(hidden_dim)
    n_layers = int(num_layers)

    input_dim = 1
    output_dim = 1

    model_select = f"stock_prediction_hidden_dim_{hidden_dim}_n_layers_{n_layers}_EPOCHS_{epoch}"

    # Instantiating the models
    model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    # Predict
    model.eval()
    model.load_state_dict(torch.load(f"{model_select}/best_stock_LSTM.pth"))
    outputs = []
    outputs_scaled = []
    start_time = time.perf_counter()

    count = 0
    while count < 5:
        print(count)
        inp = torch.from_numpy(last_sixty[:, count:count+60, :])  # torch.Size([1, 60, 1])
        # print(inp.reshape(-1))
        # labs = label
        h = model.init_hidden(1)
        out, h = model(inp.to(device).float(), h)
        last_sixty = np.append(last_sixty, out.cpu().detach().numpy().reshape(1, -1, 1)).reshape(1, -1, 1)
        # print(data_scaled.shape)
        outputs_scaled.append(out.cpu().detach().numpy())
        outputs.append(sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        # targets.append(sc.inverse_transform(labs.numpy().reshape(-1, 1)).reshape(-1))

        count += 1

    print("Evaluation Time: {}".format(str(time.perf_counter() - start_time)))
    # print(outputs_scaled)
    print(outputs)
    # print(data_scaled.shape)

    # 繪圖
    fig_list = []

    fig = plt.figure(figsize=(8, 6))
    img = plt.imread(f"{model_select}/LTSM_Prediction.png")
    plt.imshow(img)
    fig_list.append(fig)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(data[-30:])), data[-30:], color="r", label="history")
    plt.plot(np.arange(len(data[-30:]), len(data[-30:])+len(outputs)), outputs, color="b", label="Prediction")
    plt.ylabel('TAIWAN SE WEIGHTED INDEX Prediction')
    plt.legend(loc=0)
    # plt.savefig(fpath + "LTSM_Prediction.png")
    fig_list.append(fig)
    # plt.show()
    return fig_list


if __name__ == "__main__":
    print(predict())
