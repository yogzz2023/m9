import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(F.shape[0])  # Initial state covariance
        self.x = np.zeros((F.shape[0], 1))  # Initial state

    def predict(self):
        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), inv(S))
        self.x += np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

def main():
    # Read data from CSV file, read only the specified columns
    data = pd.read_csv("sample.csv", usecols=[0, 1, 2, 3])

    # Extract data into separate arrays
    measurements = data.values

    # Define state transition matrix
    F = np.eye(4)  # Assume constant velocity model for simplicity

    # Define measurement matrix
    H = np.eye(4)  # Identity matrix since measurement directly reflects state

    # Define process noise covariance matrix
    Q = np.eye(4) * 0.1  # Process noise covariance

    # Define measurement noise covariance matrix
    R = np.eye(4) * 0.01  # Measurement noise covariance, adjusted variance

    # Initialize Kalman filter
    kf = KalmanFilter(F, H, Q, R)

    # Create time array
    time = np.arange(len(measurements))

    # Lists to store predicted values
    predicted_ranges = []
    predicted_azimuths = []
    predicted_elevations = []
    predicted_times = []

    # Predict and update for each measurement
    for i, z in enumerate(measurements, start=1):
        # Predict
        kf.predict()

        # Update with measurement
        kf.update(z[:, np.newaxis])

        # Get predicted state
        predicted_state = kf.x.squeeze()

        # Extract predicted values
        predicted_ranges.append(predicted_state[0])
        predicted_azimuths.append(predicted_state[1])
        predicted_elevations.append(predicted_state[2])
        predicted_times.append(predicted_state[3])

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot measurements
    plt.plot(time, measurements[:, 0], label='Measured Range', marker='o')
    plt.plot(time, measurements[:, 1], label='Measured Azimuth', marker='o')
    plt.plot(time, measurements[:, 2], label='Measured Elevation', marker='o')
    plt.plot(time, measurements[:, 3], label='Measured Time', marker='o')

    # Plot predictions
    plt.plot(time, predicted_ranges, label='Predicted Range', linestyle='--')
    plt.plot(time, predicted_azimuths, label='Predicted Azimuth', linestyle='--')
    plt.plot(time, predicted_elevations, label='Predicted Elevation', linestyle='--')
    plt.plot(time, predicted_times, label='Predicted Time', linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Kalman Filter Prediction vs. Measurement')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
