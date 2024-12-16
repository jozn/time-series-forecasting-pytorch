import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # <--- Disables SSL verification

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
config = {
    "symbol": "IBM",  # Try "AAPL" if IBM fails
    "start": "2015-01-01",  # Fixed date range to avoid 'None data' issues
    "end": "2024-01-01",
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,  # show a date label every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # using only 1 feature (Adj Close)
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 20,
    }
}


# ------------------------------------------------------------------------------
# 2. Data downloader (using yfinance with fixed date range)
# ------------------------------------------------------------------------------
def download_data(config):
    """
    Downloads daily adjusted close prices from Yahoo Finance using yfinance
    for a fixed date range to avoid 'No data' errors.
    """
    symbol = config["symbol"]
    start_date = config["start"]
    end_date = config["end"]

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if "Adj Close" not in df.columns or len(df) == 0:
        raise ValueError(f"No data downloaded for {symbol}. Try a different ticker or date range.")

    data_close_price = df["Adj Close"].values
    data_date = df.index.strftime("%Y-%m-%d").tolist()
    num_data_points = len(data_date)
    display_date_range = f"from {data_date[0]} to {data_date[-1]}"

    print(f"Number of data points: {num_data_points} {display_date_range}")
    return data_date, data_close_price, num_data_points, display_date_range


# ------------------------------------------------------------------------------
# 3. Helper classes and functions
# ------------------------------------------------------------------------------
class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, keepdims=True)
        self.sd = np.std(x, keepdims=True)
        normalized_x = (x - self.mu) / (self.sd + 1e-8)
        return normalized_x

    def inverse_transform(self, x):
        return x * (self.sd + 1e-8) + self.mu


def prepare_data_x(x, window_size):
    """
    Creates overlapping windows of length 'window_size'.
    Returns (windowed_sequences, last_sequence_for_unseen_prediction).
    """
    n_row = x.shape[0] - window_size + 1
    if n_row <= 0:
        raise ValueError("Not enough data to create a window. Decrease window_size or check data length.")

    # Use stride_tricks to create overlapping windows
    output = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_row, window_size),
        strides=(x.strides[0], x.strides[0])
    )
    # Separate the final window as the 'unseen' slice
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    """ For each window of size W, the label is the (W+1)-th price. """
    return x[window_size:]


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        # Reshape x into [batch, seq_len, features=1]
        x = np.expand_dims(x, axis=2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


# ------------------------------------------------------------------------------
# 4. PyTorch LSTM Model
# ------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.linear_1(x)  # [batch, seq, hidden]
        x = self.relu(x)

        _, (h_n, _) = self.lstm(x)
        # h_n => [num_layers, batch, hidden_size]
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.linear_2(x)  # [batch, output_size]
        return predictions[:, -1]  # single scalar per batch


# ------------------------------------------------------------------------------
# 5. Main routine
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    data_date, data_close_price, num_data_points, display_date_range = download_data(config)

    # Plot raw close price
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [
        data_date[i] if (
                (i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) > config["plots"][
                    "xticks_interval"])
                or i == num_data_points - 1
        ) else None
        for i in range(num_data_points)
    ]
    x_ = np.arange(0, len(xticks))
    plt.xticks(x_, xticks, rotation='vertical')
    plt.title(f"Daily Adj Close price for {config['symbol']} - {display_date_range}")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()

    # Normalize
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # Prepare sequences
    window_size = config["data"]["window_size"]
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size)
    data_y = prepare_data_y(normalized_data_close_price, window_size)

    # Split dataset (train, val)
    split_index = int(len(data_y) * config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # Wrap into Datasets/Dataloaders
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    # Instantiate model
    model = LSTMModel(
        input_size=config["model"]["input_size"],
        hidden_layer_size=config["model"]["lstm_size"],
        num_layers=config["model"]["num_lstm_layers"],
        output_size=1,
        dropout=config["model"]["dropout"]
    ).to(config["training"]["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)


    # Training epoch
    def run_epoch(dataloader, is_training=False):
        epoch_loss = 0
        if is_training:
            model.train()
        else:
            model.eval()

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(config["training"]["device"])
            y_batch = y_batch.to(config["training"]["device"])

            if is_training:
                optimizer.zero_grad()

            out = model(x_batch)
            loss = criterion(out, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            batchsize = x_batch.shape[0]
            epoch_loss += (loss.detach().item() / batchsize)

        lr = scheduler.get_last_lr()[0]
        return epoch_loss, lr


    # Train loop
    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader, is_training=False)
        scheduler.step()

        print(f"Epoch[{epoch + 1}/{config['training']['num_epoch']}] | "
              f"loss train: {loss_train:.6f}, val: {loss_val:.6f} | lr: {lr_train:.6f}")

    # Plot training/validation splits
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)

    # Insert the inverse-transformed prices in their respective ranges
    to_plot_data_y_train[window_size: split_index + window_size] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + window_size:] = scaler.inverse_transform(data_y_val)

    # Replace zeros with None for better plotting
    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [
        data_date[i] if (
                (i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) > config["plots"][
                    "xticks_interval"])
                or i == num_data_points - 1
        ) else None
        for i in range(num_data_points)
    ]
    x_ = np.arange(0, len(xticks))
    plt.xticks(x_, xticks, rotation='vertical')
    plt.title(f"Daily Adj Close prices for {config['symbol']} - training/validation split")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # Predict next day using the unseen final window
    model.eval()
    x_unseen = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    prediction = model(x_unseen)
    prediction = prediction.cpu().detach().numpy()
    predicted_next_day = scaler.inverse_transform(prediction)

    print("Predicted close price for the next trading day:", round(predicted_next_day[0], 2))

    print("All done.")
