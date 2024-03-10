import numpy as np
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
    # Sample Measurement
    measurement_range = 94779.54  # Sample measurement range
    measurement_azimuth = 217.0574  # Sample measurement azimuth
    measurement_elevation = 2.7189  # Sample measurement elevation
    measurement_time = 21486.916  # Sample measurement time

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

    # Predict
    kf.predict()

    # Update with measurement
    z = np.array([[measurement_range], [measurement_azimuth], [measurement_elevation], [measurement_time]])
    kf.update(z)

    # Get predicted state
    predicted_state = kf.x

    # Extract predicted range, azimuth, elevation, and time
    predicted_range = predicted_state[0][0]
    predicted_azimuth = predicted_state[1][0]
    predicted_elevation = predicted_state[2][0]
    predicted_time = predicted_state[3][0]

    print("Predicted Range:", predicted_range)
    print("Predicted Azimuth:", predicted_azimuth)
    print("Predicted Elevation:", predicted_elevation)
    print("Predicted Time:", predicted_time)

if __name__ == "__main__":
    main()
