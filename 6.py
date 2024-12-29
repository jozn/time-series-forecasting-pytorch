import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# -------------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------------
config = {
    "symbol": "AAPL_MOCK",
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
        "num_data_points": 1000  # Total number of mock daily points
    },
    "plots": {
        "xticks_interval": 50,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_test": "#FF4136",
        "color_normalized": "#FF851B",
        "color_window": "#B10DC9",
        "color_target": "#39CCCC",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # or "cuda"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 20,
    },
    "trading_simulation": {
        "lookback_days": 100
    }
}

# -------------------------------------------------------------------------------
# 2. Generate Mock Data
# -------------------------------------------------------------------------------
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
    data = np.clip(data, a_min=1.0, a_max=None)  # Ensure positivity

    # Generate mock date strings
    dates = [f"2020-01-01 + {i}d" for i in range(num_points)]
    return dates, data

# -------------------------------------------------------------------------------
# 3. Helper Classes & Functions
# -------------------------------------------------------------------------------
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
    Prepare input sequences for the model using a sliding window approach.
    """
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0])
    )
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    """
    Prepare target values for the model.
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

# -------------------------------------------------------------------------------
# 4. LSTM Model in PyTorch Lightning
# -------------------------------------------------------------------------------
class LSTMModelPL(pl.LightningModule):
    def __init__(
        self,
        input_size=1,
        hidden_layer_size=32,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        lr=0.01,
        scheduler_step=20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.scheduler_step = scheduler_step

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.criterion = nn.MSELoss()

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
        batch_size = x.size(0)
        x = self.linear_1(x)
        x = self.relu(x)
        _, (h_n, _) = self.lstm(x)
        x = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        outputs = self.forward(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        outputs = self.forward(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.1)
        return [optimizer], [scheduler]

# -------------------------------------------------------------------------------
# 5. Plotting Helpers
# -------------------------------------------------------------------------------
def plot_data(dates, data, title, color, label, xticks_interval):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, data, color=color, label=label)
    plt.xticks(rotation='vertical')
    plt.title(title)
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

def plot_prices(dates, actual, train, val, config):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, actual, color=config["plots"]["color_actual"], label="Mock Prices")
    plt.plot(dates, train, label="Prices (Train)", color=config["plots"]["color_train"])
    plt.plot(dates, val, label="Prices (Validation)", color=config["plots"]["color_val"])

    xticks = [
        dates[i] if (i % config["plots"]["xticks_interval"] == 0 or i == len(dates) - 1)
        else None for i in range(len(dates))
    ]
    xvals = np.arange(len(dates))
    plt.xticks(xvals, xticks, rotation='vertical')

    plt.title("Mock Daily Close Price for {} - Total Points: {}".format(
        config['symbol'], len(dates)))
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------------
# 6. Main with Extended Trading Simulation
# -------------------------------------------------------------------------------
def main():
    seed_everything(42)

    # ---------------------------------------------------------------------------
    # Generate Synthetic Data
    # ---------------------------------------------------------------------------
    num_data_points = config["data"]["num_data_points"]
    data_dates, data_close_price = generate_mock_data(num_data_points)

    # Plot Raw Mock Prices
    plot_data(
        dates=data_dates,
        data=data_close_price,
        title=f"Raw Mock Daily Close Price for {config['symbol']} - Total Points: {num_data_points}",
        color=config["plots"]["color_actual"],
        label="Mock Prices",
        xticks_interval=config["plots"]["xticks_interval"]
    )

    # ---------------------------------------------------------------------------
    # Normalize Data
    # ---------------------------------------------------------------------------
    scaler = Normalizer()
    normalized_close_price = scaler.fit_transform(data_close_price)

    # Plot Normalized Data
    plot_data(
        dates=data_dates,
        data=normalized_close_price,
        title=f"Normalized Close Price for {config['symbol']}",
        color=config["plots"]["color_normalized"],
        label="Normalized Prices",
        xticks_interval=config["plots"]["xticks_interval"]
    )

    # ---------------------------------------------------------------------------
    # Prepare Data (Windowing)
    # ---------------------------------------------------------------------------
    window_size = config["data"]["window_size"]
    data_x, data_x_unseen = prepare_data_x(normalized_close_price, window_size)
    data_y = prepare_data_y(normalized_close_price, window_size)

    # Plot Input Windows (First 20 Windows)
    plt.figure(figsize=(12, 6))
    for i in range(20):
        plt.plot(range(window_size), data_x[i], label=f"Window {i + 1}")
    plt.title("Sample Input Windows")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Targets
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_dates[window_size:], data_y, color=config["plots"]["color_target"], label="Target Prices")
    plt.title("Target Prices After Windowing")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.xticks(rotation='vertical')
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Train/Validation Split
    # ---------------------------------------------------------------------------
    split_index = int(len(data_y) * config["data"]["train_split_size"])
    data_x_train, data_x_val = data_x[:split_index], data_x[split_index:]
    data_y_train, data_y_val = data_y[:split_index], data_y[split_index:]

    # Plot Train vs Validation Split
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_dates[window_size:split_index + window_size], data_y_train,
             color=config["plots"]["color_train"], label="Train Targets")
    plt.plot(data_dates[split_index + window_size:], data_y_val,
             color=config["plots"]["color_val"], label="Validation Targets")
    plt.title("Train and Validation Targets Split")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.xticks(rotation='vertical')
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Create Datasets and DataLoaders
    # ---------------------------------------------------------------------------
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    train_loader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(dataset_val,   batch_size=config["training"]["batch_size"], shuffle=False)

    print("Train data shape:", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape:", dataset_val.x.shape, dataset_val.y.shape)

    # ---------------------------------------------------------------------------
    # Initialize Model & Trainer
    # ---------------------------------------------------------------------------
    device_str = config["training"]["device"]
    model_pl = LSTMModelPL(
        input_size=config["model"]["input_size"],
        hidden_layer_size=config["model"]["lstm_size"],
        num_layers=config["model"]["num_lstm_layers"],
        output_size=1,
        dropout=config["model"]["dropout"],
        lr=config["training"]["learning_rate"],
        scheduler_step=config["training"]["scheduler_step_size"]
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epoch"],
        accelerator=device_str,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False
    )

    trainer.fit(model_pl, train_loader, val_loader)

    # ---------------------------------------------------------------------------
    # Plot Training and Validation Data in Original Scale
    # ---------------------------------------------------------------------------
    to_plot_train = np.full(num_data_points, np.nan)
    to_plot_val = np.full(num_data_points, np.nan)

    to_plot_train[window_size:split_index + window_size] = scaler.inverse_transform(data_y_train)
    to_plot_val[split_index + window_size:] = scaler.inverse_transform(data_y_val)

    plot_prices(data_dates, data_close_price, to_plot_train, to_plot_val, config)

    # ---------------------------------------------------------------------------
    # Predict Next Day's Price (single-step)
    # ---------------------------------------------------------------------------
    model_pl.eval()
    with torch.no_grad():
        x_unseen_tensor = torch.tensor(data_x_unseen).float().unsqueeze(0).unsqueeze(2)
        prediction = model_pl(x_unseen_tensor)
        prediction_val = prediction.numpy()
        predicted_next_day = scaler.inverse_transform(prediction_val)

    print("Predicted close price for the next 'trading' day:", round(predicted_next_day[0], 2))

    # ---------------------------------------------------------------------------
    # Plot the Next Day Prediction
    # ---------------------------------------------------------------------------
    extended_dates = data_dates + ["Next_Day_Prediction"]
    extended_prices = np.append(data_close_price, predicted_next_day[0])

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_dates, data_close_price, color=config["plots"]["color_actual"], label="Mock Prices")
    plt.plot(data_dates, to_plot_train, color=config["plots"]["color_train"], label="Train Targets")
    plt.plot(data_dates, to_plot_val, color=config["plots"]["color_val"], label="Validation Targets")
    plt.plot(extended_dates[-2:], extended_prices[-2:], color=config["plots"]["color_pred_test"],
             marker='o', label="Next Day Prediction")

    xticks = [
        extended_dates[i] if (i % config["plots"]["xticks_interval"] == 0 or i == len(extended_dates) - 1)
        else None for i in range(len(extended_dates))
    ]
    xvals = np.arange(len(extended_dates))
    plt.xticks(xvals, xticks, rotation='vertical')

    plt.title("Mock Daily Close Price with Next Day Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Extended Simulation: Day Trading Over the Last 100 Days
    # ---------------------------------------------------------------------------
    simulate_days = config["trading_simulation"]["lookback_days"]
    start_index = len(data_y) - simulate_days

    total_trades = 0
    total_wins = 0

    print("\nSimulation for the last", simulate_days, "days:")

    for i in range(start_index, len(data_y) - 1):
        current_window = normalized_close_price[i: i + window_size]
        current_price_actual = data_close_price[i]
        next_day_actual = data_close_price[i + 1]

        window_tensor = torch.tensor(current_window).float().unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred_norm = model_pl(window_tensor)
            pred_norm_np = pred_norm.numpy()
        predicted_next_day_price = scaler.inverse_transform(pred_norm_np)[0]

        direction = "LONG" if (predicted_next_day_price > current_price_actual) else "SHORT"
        actual_move = next_day_actual - current_price_actual

        is_win = False
        if direction == "LONG" and actual_move > 0:
            is_win = True
        elif direction == "SHORT" and actual_move < 0:
            is_win = True

        total_trades += 1
        if is_win:
            total_wins += 1
        current_winrate = total_wins / total_trades

        print(f"Day Index {i} => Predicted Next: {predicted_next_day_price:.2f} | "
              f"Actual Next: {next_day_actual:.2f} | "
              f"Direction: {direction} | "
              f"Result: {'WIN' if is_win else 'LOSS'} | "
              f"Running Win Rate: {current_winrate:.2%}")

    print("\nFinal Simulation Results:")
    print("Total Trades:", total_trades)
    print("Total Wins:", total_wins)
    print("Final Win Rate:", f"{(total_wins / total_trades) * 100:.2f}%")

    print("\nAll done.")

if __name__ == "__main__":
    main()
