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

df = pd.read_csv("歷年台股大盤指數.csv", index_col=0)
# print(df.info())
df["Open"] = df["Open"].apply(lambda x: x.replace(",", "")).astype("float")
# print(df["Open"])

sc = MinMaxScaler(feature_range=(0, 1))
# sc = StandardScaler()
dataset_scaled = sc.fit_transform(np.array(df["Open"].values).reshape(-1, 1))
print(dataset_scaled.shape)

# training_set = df.iloc[:, 2:3].values
training_set = dataset_scaled[:-180, :]
# print(training_set.shape)

training_set = np.array(training_set)
# print(training_set.shape)

X_train = []
y_train = []
for i in range(60, training_set.shape[0]):
    X_train.append(training_set[i - 60:i, 0])
    y_train.append(training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(f"X_train: {X_train.shape}")  # X_train: (5877, 60, 1)
print(f"y_train: {y_train.shape}")  # y_train: (5877,)

# --------------
testing_set = dataset_scaled[-180:, :]
# print(testing_set.shape)

testing_set = np.array(testing_set)
# print(testing_set.shape)

X_test = []
y_test = []
for i in range(60, testing_set.shape[0]):
    X_test.append(testing_set[i - 60:i, 0])
    y_test.append(testing_set[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(f"X_test: {X_test.shape}")  # X_test: (120, 60, 1)
print(f"y_test: {y_test.shape}")  # y_test: (120,)

batch_size = 256

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
# print(next(iter(train_loader)))

test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_data, shuffle=False, batch_size=X_test.shape[0], drop_last=True)
# print(next(iter(test_loader)))

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
        # print(hidden[0].shape, hidden[1].shape)  # torch.Size([1, 120, 512]) torch.Size([1, 120, 512])
        return hidden


def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=1000):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]  # input_dim: 1
    print(f"input_dim: {input_dim}")
    output_dim = 1
    n_layers = 1
    # Instantiating the models

    model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    epoch_times = []
    train_losses = []
    best_loss = np.Inf
    PATH = 'stock_LSTM.pth'
    directory = f"stock_prediction_hidden_dim_{hidden_dim}_n_layers_{n_layers}_EPOCHS_{EPOCHS}/"
    # Start training loop

    for epoch in range(1, EPOCHS + 1):
        # if 0 < epoch <= 50:
        #     learn_rate = learn_rate
        # elif 50 < epoch <= 100:
        #     learn_rate = 0.0005
        # else:
        #     learn_rate = 0.00025
        #
        # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        start_time = time.perf_counter()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        train_loss = 0.
        counter = 0

        for x, label in train_loader:
            # print(x.shape)
            # print(label.shape)
            counter += 1
            h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            train_loss += loss.item() * x.size(0)

        current_time = time.perf_counter()
        print("Epoch {}/{} Done, Total Loss: {}, LR: {}".format(epoch, EPOCHS, avg_loss / len(train_loader), learn_rate))
        print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 模型儲存
        if not os.path.exists(directory):
            os.makedirs(directory)

        if train_loss < best_loss:
            best_loss = train_loss
            print(f"saving model.....\npresent best loss: {best_loss}")
            torch.save(model.state_dict(), directory + f"best_loss_{best_loss:.5f}@{epoch}_" + PATH)


    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model, train_losses, directory


# 使用對稱平均絕對百分比誤差(sMAPE)來評估模型。
# sMAPE是預測值和實際值之間的絕對差值之和除以預測值和實際值的平均值，因此給出了衡量誤差量的百分比。
def evaluate(model, data_loader):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for x, label in data_loader:
        # print(x.shape)
        # print(label.shape)
        inp = x  # inp shape: torch.Size([120, 60, 1])
        labs = label
        # print(f"inp shape: {inp.shape}")
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(sc.inverse_transform(labs.numpy().reshape(-1, 1)).reshape(-1))
    print("Evaluation Time: {}".format(str(time.perf_counter() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE


lr = 0.001

lstm_model, losses, fpath = train(train_loader, lr)

lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_loader)

# print(lstm_outputs)
# print(len(lstm_outputs[0]))
# print(targets)
# print(len(targets[0]))

lstm_outputs_x_axis = np.arange(len(lstm_outputs[0]))
targets_x_axis = np.arange(len(targets[0]))


plt.figure(figsize=(14, 10))
# plt.subplot(2, 2, 1)
plt.plot(lstm_outputs_x_axis, lstm_outputs[0], color="r", label="LSTM Predicted")
plt.plot(targets_x_axis, targets[0], color="b", label="Actual")
plt.ylabel('TAIWAN SE WEIGHTED INDEX')
plt.legend(loc=0)
plt.savefig(fpath + "LTSM_Prediction.png")

plt.show()

plt.figure(figsize=(14, 10))
# plt.subplot(2, 2, 1)
plt.plot(losses, color="g", label="Loss")
plt.ylabel('Loss')
plt.legend(loc=0)
plt.savefig(fpath + "loss.png")

plt.show()
