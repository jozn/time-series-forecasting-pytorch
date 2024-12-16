import numpy as np
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
    "symbol": "AAPL_MOCK",
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
        "num_data_points": 1000   # total number of mock daily points
    },
    "plots": {
        "xticks_interval": 50,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",      # or "cuda"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 20,
    }
}


# ------------------------------------------------------------------------------
# 2. Generate mock data
# ------------------------------------------------------------------------------
def generate_mock_data(num_points=1000):
    """
    Creates a synthetic time series resembling a stock price:
    Random walk + sinusoidal pattern.
    """
    np.random.seed(42)

    base_price = 150.0
    random_walk = np.random.normal(loc=0.0, scale=1.0, size=num_points).cumsum()
    x = np.linspace(0, 4 * np.pi, num_points)
    sinusoid = 10.0 * np.sin(x)

    data = base_price + random_walk * 0.5 + sinusoid
    data = data.clip(min=1.0)  # ensure positivity

    # mock date strings
    dates = [f"2020-01-01 + {i}d" for i in range(num_points)]
    return dates, data


# ------------------------------------------------------------------------------
# 3. Helper classes & functions
# ------------------------------------------------------------------------------
class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, keepdims=True)
        self.sd = np.std(x, keepdims=True)
        return (x - self.mu) / (self.sd + 1e-8)

    def inverse_transform(self, x):
        return x * (self.sd + 1e-8) + self.mu


def prepare_data_x(x, window_size):
    """
    Return all overlapping windows except the last as `data_x`
    and the last window as 'data_x_unseen'.
    """
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0])
    )
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    """
    The label for each window is the next day's price.
    """
    return x[window_size:]


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, axis=2)  # [batch, sequence, features=1]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


# ------------------------------------------------------------------------------
# 4. LSTM Model
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

        _, (h_n, _) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.linear_2(x)  # shape => [batch, output_size]
        return predictions[:, -1]


# ------------------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    num_data_points = config["data"]["num_data_points"]
    data_date, data_close_price = generate_mock_data(num_data_points)

    # Plot raw mock prices
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"], label="Mock prices")
    plt.xticks(rotation='vertical')
    plt.title(f"Mock daily close price for {config['symbol']} - total points: {num_data_points}")
    plt.grid(which='major', axis='y', linestyle='--')  # removed b=None
    plt.legend()
    plt.show()

    # Normalize
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # Prepare data (windowing)
    window_size = config["data"]["window_size"]
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size)
    data_y = prepare_data_y(normalized_data_close_price, window_size)

    # Train/validation split
    split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val   = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val   = data_y[split_index:]

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val   = TimeSeriesDataset(data_x_val, data_y_val)

    train_loader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(dataset_val,   batch_size=config["training"]["batch_size"], shuffle=False)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    # Model
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

    # Train
    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_loader, is_training=True)
        loss_val, _ = run_epoch(val_loader, is_training=False)
        scheduler.step()

        print(f"Epoch[{epoch+1}/{config['training']['num_epoch']}] "
              f"| loss train:{loss_train:.6f}, val:{loss_val:.6f} "
              f"| lr:{lr_train:.6f}")

    # Plot training/validation split
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val   = np.zeros(num_data_points)
    to_plot_data_y_train[window_size : split_index + window_size] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + window_size :] = scaler.inverse_transform(data_y_val)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val   = np.where(to_plot_data_y_val   == 0, None, to_plot_data_y_val)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val,   label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [
        data_date[i] if (i % config["plots"]["xticks_interval"] == 0 or i == num_data_points-1)
        else None for i in range(num_data_points)
    ]
    xvals = np.arange(0, len(xticks))
    plt.xticks(xvals, xticks, rotation='vertical')
    plt.title("Training vs Validation (Mock Data)")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # Predict next day's price using the last unseen window
    model.eval()
    x_unseen = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    prediction = model(x_unseen)
    prediction_val = prediction.cpu().detach().numpy()
    predicted_next_day = scaler.inverse_transform(prediction_val)

    print("Predicted close price for the next 'trading' day:", round(predicted_next_day[0], 2))
    print("All done.")
