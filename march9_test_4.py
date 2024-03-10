import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from ipywidgets import interact, widgets

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

def main(variable='Range'):
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

    # Create time array (assuming time is in milliseconds)
    time = np.arange(len(measurements)) * 10  # assuming time is in milliseconds

    # Lists to store predicted values
    predicted_values = []

    # Predict and update for each measurement
    for i, z in enumerate(measurements, start=1):
        # Predict
        kf.predict()

        # Update with measurement
        kf.update(z[:, np.newaxis])

        # Get predicted state
        predicted_state = kf.x.squeeze()

        # Append predicted value based on selected variable
        if variable == 'Range':
            predicted_values.append(predicted_state[0])
        elif variable == 'Azimuth':
            predicted_values.append(predicted_state[1])
        elif variable == 'Elevation':
            predicted_values.append(predicted_state[2])
        elif variable == 'Time':
            predicted_values.append(predicted_state[3])

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot measurements
    plt.plot(time, measurements[:, ['Range', 'Azimuth', 'Elevation', 'Time'].index(variable)], label='Measured ' + variable, marker='o')

    # Plot predictions
    plt.plot(time, predicted_values, label='Predicted ' + variable, linestyle='--')

    plt.xlabel('Time (ms)')
    plt.ylabel(variable)
    plt.title('Kalman Filter Prediction vs. Measurement')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create dropdown widgets for each variable
variable_dropdown_range = widgets.Dropdown(
    options=['Range', 'Azimuth', 'Elevation', 'Time'],
    value='Range',
    description='Variable:',
    disabled=False,
)

variable_dropdown_azimuth = widgets.Dropdown(
    options=['Range', 'Azimuth', 'Elevation', 'Time'],
    value='Azimuth',
    description='Variable:',
    disabled=False,
)

variable_dropdown_elevation = widgets.Dropdown(
    options=['Range', 'Azimuth', 'Elevation', 'Time'],
    value='Elevation',
    description='Variable:',
    disabled=False,
)

variable_dropdown_time = widgets.Dropdown(
    options=['Range', 'Azimuth', 'Elevation', 'Time'],
    value='Time',
    description='Variable:',
    disabled=False,
)

# Interact function to update plot based on dropdown selection
interact(main, variable=variable_dropdown_range)
interact(main, variable=variable_dropdown_azimuth)
interact(main, variable=variable_dropdown_elevation)
interact(main, variable=variable_dropdown_time)