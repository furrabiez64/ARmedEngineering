import time
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations
import pandas as pd
import serial  # Fix for NameError: name 'serial' is not defined
import serial.tools.list_ports  # For automatic port detection


# Check available ports
def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise Exception("No serial ports found. Please connect your EEG device.")
    print("Available Serial Ports:")
    for i, port in enumerate(ports):
        print(f"{i}: {port.device}")
    selection = int(input("Select the port number for your EEG device: "))
    return ports[selection].device


# -------- SETUP PARAMETERS --------
board_id = 57  # NeuroPawn Knightboard
params = BrainFlowInputParams()
params.serial_port = select_serial_port()  # Dynamically detected

# Start session
BoardShim.enable_dev_board_logger()
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# Get channel and sampling info
eeg_channels = BoardShim.get_eeg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
window_size = 2  # seconds
num_samples = window_size * sampling_rate

# -------- FUNCTION TO COMPUTE ALPHA/BETA RATIO --------
def compute_attention(channel_data):
    def band_power(data, low, high):
        psd, freqs = DataFilter.get_psd_welch(
            data, nfft=256, overlap=128, sampling_rate=sampling_rate,
            window=WindowOperations.HANNING.value
        )
        band_indices = np.where((freqs >= low) & (freqs <= high))
        return np.sum(psd[band_indices])

    alpha = band_power(channel_data, 8.0, 13.0)
    beta = band_power(channel_data, 13.0, 30.0)

    # Prevent division by zero
    return beta / alpha if alpha > 0 else 0

# -------- REAL-TIME LOOP --------
print("Starting real-time attention monitoring...\nPress Ctrl+C to stop.")
log_df = pd.DataFrame(columns=["Timestamp", "Attention"])
try:
    while True:
        # Wait to accumulate enough data
        time.sleep(window_size)

        # Get recent data window
        data = board.get_current_board_data(num_samples)
        attention_levels = []

        # Compute attention level for each EEG channel
        for ch in eeg_channels:
            ch_data = data[ch]
            attention = compute_attention(ch_data)
            attention_levels.append(attention)

        # You can take the average, or use just one channel
        avg_attention = np.mean(attention_levels)
        print(f"🧠 Attention Level (beta/alpha): {avg_attention:.2f}")

        # TODO: Send `avg_attention` to AR headset software (e.g., over WebSocket, MQTT, shared file, etc.)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    board.stop_stream()
    board.release_session()
    log_df.to_excel("attention_log.xlsx", index=False)
    # Save Excel log
    log_df.to_excel("attention_log.xlsx", index=False)
    print("✅ Log saved to attention_log.xlsx")

    # Plot results
    log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"])
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["Timestamp"], log_df["Attention"], marker='o')
    plt.title("Attention Over Time")
    plt.xlabel("Time")
    plt.ylabel("Beta/Alpha Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

