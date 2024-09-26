import numpy as np

# Extended Kalman Filter: Correction of the state estimate using measurements
def ekf_update(state, P, measurement, measurement_pred, H, R):
    """
    Extended Kalman Filter update step for error correction using new sensor measurements.
    
    Args:
        state (np.array): Predicted state [x, y, theta].
        P (np.array): Predicted covariance matrix.
        measurement (np.array): Actual measurement from sensors.
        measurement_pred (np.array): Predicted measurement from the state.
        H (np.array): Measurement Jacobian (linearized observation model).
        R (np.array): Measurement noise covariance matrix.
    
    Returns:
        np.array: Updated state [x, y, theta].
        np.array: Updated covariance matrix.
    """
    # Innovation (difference between actual measurement and predicted measurement)
    innovation = measurement - measurement_pred
    innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))  # Normalize the angle (for theta)

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Update the state estimate
    state = state + K @ innovation

    # Normalize theta
    state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))

    # Update the covariance estimate
    P = (np.eye(len(state)) - K @ H) @ P

    return state, P

# Prediction for measurement (landmark position from SLAM or sensor data)
def measurement_prediction(state, landmark):
    """
    Predict the measurement (range and bearing) to a landmark given the current state.
    
    Args:
        state (np.array): Current state [x, y, theta].
        landmark (np.array): Position of the landmark [x_l, y_l].

    Returns:
        np.array: Predicted measurement [range, bearing].
    """
    x, y, theta = state
    x_l, y_l = landmark

    # Calculate range and bearing to the landmark
    delta_x = x_l - x
    delta_y = y_l - y
    r = np.sqrt(delta_x**2 + delta_y**2)  # Range
    phi = np.arctan2(delta_y, delta_x) - theta  # Bearing (relative to robot)

    return np.array([r, phi])

# Main function for EKF correction based on sensor measurements
def run_ekf_correction(state, P, measurements, landmarks, R):
    """
    Run the EKF correction step to adjust the state and covariance estimates.
    
    Args:
        state (np.array): Current state [x, y, theta].
        P (np.array): Current covariance matrix.
        measurements (list): List of sensor measurements [range, bearing].
        landmarks (list): List of known landmarks [x_l, y_l] corresponding to the measurements.
        R (np.array): Measurement noise covariance matrix.
    
    Returns:
        np.array: Corrected state [x, y, theta].
        np.array: Corrected covariance matrix.
    """
    for i, landmark in enumerate(landmarks):
        # Get the actual sensor measurement
        measurement = measurements[i]

        # Predict what the measurement should be based on the current state and landmark
        measurement_pred = measurement_prediction(state, landmark)

        # Define the Jacobian H (linearized observation model)
        H = np.array([[1, 0, 0],  # Simplified Jacobian for this example
                      [0, 1, 0]])

        # Run EKF update for the current landmark and measurement
        state, P = ekf_update(state, P, measurement, measurement_pred, H, R)
    
    return state, P

# Simulate sensor measurements (range and bearing to landmarks)
def simulate_measurements(state, landmarks):
    """
    Simulate sensor measurements (range and bearing) to known landmarks.
    
    Args:
        state (np.array): Current robot state [x, y, theta].
        landmarks (list): List of known landmarks [x_l, y_l].

    Returns:
        list: List of simulated measurements [range, bearing].
    """
    measurements = []
    for landmark in landmarks:
        measurement = measurement_prediction(state, landmark)
        measurements.append(measurement)
    
    return measurements

# Main function to run error correction using EKF
if __name__ == "__main__":
    # Initialize the robot's state [x, y, theta] and covariance matrix P
    initial_state = np.array([0.0, 0.0, 0.0])  # Start at origin
    P = np.eye(3) * 0.1  # Initial uncertainty in the state
    
    # Define known landmarks in the environment (example [x_l, y_l])
    landmarks = np.array([[5.0, 5.0], [10.0, 0.0], [7.0, 8.0]])

    # Simulate the actual sensor measurements (range and bearing to landmarks)
    true_state = np.array([6.0, 4.0, 0.5])  # True state of the robot for simulation
    measurements = simulate_measurements(true_state, landmarks)

    # Define the measurement noise covariance matrix
    R = np.diag([0.1, np.deg2rad(5)])  # Measurement noise in range and bearing

    # Run EKF correction to update state and covariance based on measurements
    corrected_state, corrected_P = run_ekf_correction(initial_state, P, measurements, landmarks, R)

    print("Corrected state after EKF:", corrected_state)
    print("Corrected covariance matrix P after EKF:\n", corrected_P)
