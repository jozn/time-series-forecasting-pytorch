import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from alpha_vantage.timeseries import TimeSeries


# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
config = {
    "alpha_vantage": {
        # "key": "YOUR_API_KEY",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "key": "XR6O2HXDXKDO85D1",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,  # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature (close price)
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}


# ------------------------------------------------------------------------------
# 2. Data downloader
# ------------------------------------------------------------------------------
def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(
        config["alpha_vantage"]["symbol"],
        outputsize=config["alpha_vantage"]["outputsize"]
    )

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [
        float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()
    ]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range


# ------------------------------------------------------------------------------
# 3. Helper classes and functions (Normalizer, Dataset, windowing, etc.)
# ------------------------------------------------------------------------------
class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        # For a single feature, reshape x into [batch, sequence, features] for LSTM
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def prepare_data_x(x, window_size):
    # Perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0])
    )
    # Separate final row as the 'unseen' slice
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # Use the next day as the label
    output = x[window_size:]
    return output


# ------------------------------------------------------------------------------
# 4. Model definition
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
        x = self.linear_1(x)
        x = self.relu(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        # Reshape hidden output to [batch, features]
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.linear_2(x)
        # Return last output
        return predictions[:, -1]


# ------------------------------------------------------------------------------
# 5. Main routine: data loading, preparation, model training, plots
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 5.1 Download raw data
    data_date, data_close_price, num_data_points, display_date_range = download_data(config)

    # 5.2 Plot raw close price
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [
        data_date[i] if (
            (i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) > config["plots"]["xticks_interval"])
            or i == num_data_points - 1
        ) else None
        for i in range(num_data_points)
    ]
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()

    # 5.3 Normalize the data
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # 5.4 Prepare windowed data
    data_x, data_x_unseen = prepare_data_x(
        normalized_data_close_price,
        window_size=config["data"]["window_size"]
    )
    data_y = prepare_data_y(
        normalized_data_close_price,
        window_size=config["data"]["window_size"]
    )

    # 5.5 Split dataset
    split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # 5.6 Prepare datasets and dataloaders
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val   = TimeSeriesDataset(data_x_val, data_y_val)
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader   = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    # 5.7 Initialize model, criterion, optimizer, scheduler
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

    # 5.8 Define training/validation step
    def run_epoch(dataloader, is_training=False):
        epoch_loss = 0
        if is_training:
            model.train()
        else:
            model.eval()

        for idx, (x_batch, y_batch) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()

            batchsize = x_batch.shape[0]
            x_batch = x_batch.to(config["training"]["device"])
            y_batch = y_batch.to(config["training"]["device"])

            out = model(x_batch)
            loss = criterion(out.contiguous(), y_batch.contiguous())

            if is_training:
                loss.backward()
                optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = scheduler.get_last_lr()[0]
        return epoch_loss, lr

    # 5.9 Train the model
    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader, is_training=False)
        scheduler.step()

        print('Epoch[{}/{}] | loss train:{:.6f}, val:{:.6f} | lr:{:.6f}'
              .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

    # 5.10 Plot training/validation actual prices (unpredicted, just raw splits)
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val   = np.zeros(num_data_points)
    window_size          = config["data"]["window_size"]

    to_plot_data_y_train[window_size : split_index + window_size] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + window_size :] = scaler.inverse_transform(data_y_val)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val   = np.where(to_plot_data_y_val   == 0, None, to_plot_data_y_val)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val,   label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [
        data_date[i] if (
            (i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) > config["plots"]["xticks_interval"])
            or i == num_data_points - 1
        ) else None
        for i in range(num_data_points)
    ]
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - training/validation split")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # 5.11 Make a final prediction (for the next day beyond the last known day)
    # Example: creating a dummy unseen input from the final window
    # If 'data_x_unseen' was extracted from prepare_data_x, we can predict directly:
    model.eval()
    x_unseen = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    prediction = model(x_unseen)
    prediction = prediction.cpu().detach().numpy()
    predicted_next_day = scaler.inverse_transform(prediction)

    print("Predicted close price for the next trading day:", round(predicted_next_day[0], 2))

    # 5.12 Example of plotting the last few days + next-day prediction
    # (Assume we had 'predicted_val' from validation predictions, otherwise skip)
    # For illustration, define dummy arrays if needed:
    # predicted_val = out_of_sample_predictions_for_validation  # if you had them

    # Below is a hypothetical code snippet if you had 'predicted_val' in memory.
    # You can adapt it as needed:

    """
    plot_range = 10
    # Suppose 'predicted_val' and data_y_val exist
    # We'll fake an array just for demonstration
    predicted_val = np.zeros_like(data_y_val)  # Replace with actual predicted values

    to_plot_data_y_val_p = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val_p[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]
    to_plot_data_y_test_pred[plot_range-1] = predicted_next_day

    to_plot_data_y_val_p = np.where(to_plot_data_y_val_p == 0, None, to_plot_data_y_val_p)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    plot_date_test = data_date[-(plot_range - 1):]
    plot_date_test.append("tomorrow")

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val_p, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted next-day price", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
    plt.title("Predicted close price of the next trading day")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
    """

    print("All done.")
