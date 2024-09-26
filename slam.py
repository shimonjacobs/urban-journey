import numpy as np

# Initialize map and robot state
def initialize_slam(grid_size=100, grid_resolution=1.0):
    """
    Initialize SLAM state with an empty occupancy grid and initial robot pose.
    
    Args:
        grid_size (int): The size of the occupancy grid (grid_size x grid_size).
        grid_resolution (float): Resolution of each grid cell in meters.
    
    Returns:
        map (np.array): Occupancy grid initialized with zeros.
        state (np.array): Initial robot state [x, y, theta].
        P (np.array): Initial covariance matrix.
    """
    # Initialize an empty occupancy grid for mapping
    occupancy_grid = np.zeros((grid_size, grid_size))
    
    # Initial state vector [x, y, theta] (starting position at the center of the grid)
    initial_state = np.array([grid_size / 2 * grid_resolution, grid_size / 2 * grid_resolution, 0.0])
    
    # Covariance matrix (initial uncertainty)
    initial_P = np.eye(3) * 0.1  # Small initial uncertainty
    
    return occupancy_grid, initial_state, initial_P


# Motion model: predict next state based on control input
def predict_state(state, control_input, dt):
    """
    Predict the next state using the motion model.
    
    Args:
        state (np.array): Current state [x, y, theta].
        control_input (np.array): Control input [v, omega] (linear and angular velocities).
        dt (float): Time step.

    Returns:
        np.array: Predicted next state [x, y, theta].
    """
    x, y, theta = state
    v, omega = control_input  # Linear and angular velocity

    # Update position and orientation based on control input
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    # Normalize theta to stay within [-pi, pi]
    theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))

    return np.array([x_next, y_next, theta_next])


# EKF: Predict step for covariance
def predict_covariance(P, control_input, Q, dt):
    """
    Predict the next covariance using the motion model and noise.
    
    Args:
        P (np.array): Current covariance matrix.
        control_input (np.array): Control input [v, omega].
        Q (np.array): Process noise covariance matrix.
        dt (float): Time step.

    Returns:
        np.array: Predicted covariance matrix.
    """
    v, omega = control_input

    # Jacobian of the motion model with respect to the state
    F = np.array([[1, 0, -v * np.sin(P[2]) * dt],
                  [0, 1, v * np.cos(P[2]) * dt],
                  [0, 0, 1]])

    # Update the covariance matrix
    P_next = F @ P @ F.T + Q

    return P_next


# Observation model: Map feature detection (e.g., LIDAR)
def observation_model(state, landmark_pos):
    """
    Predict the observation (range and bearing) to a landmark given the state.
    
    Args:
        state (np.array): Current state [x, y, theta].
        landmark_pos (np.array): Position of the landmark [x_l, y_l].

    Returns:
        np.array: Observation [range, bearing] to the landmark.
    """
    x, y, theta = state
    x_l, y_l = landmark_pos

    # Range and bearing from robot to landmark
    delta_x = x_l - x
    delta_y = y_l - y
    r = np.sqrt(delta_x ** 2 + delta_y ** 2)
    phi = np.arctan2(delta_y, delta_x) - theta  # Bearing relative to robot

    return np.array([r, phi])


# EKF: Update step for state and covariance based on new observation
def update_state(state, P, z, z_pred, H, R):
    """
    Update the state and covariance matrix based on a new observation.
    
    Args:
        state (np.array): Predicted state [x, y, theta].
        P (np.array): Predicted covariance matrix.
        z (np.array): Actual observation [range, bearing].
        z_pred (np.array): Predicted observation [range, bearing].
        H (np.array): Jacobian of the observation model.
        R (np.array): Measurement noise covariance matrix.

    Returns:
        np.array: Updated state [x, y, theta].
        np.array: Updated covariance matrix.
    """
    # Innovation (difference between actual and predicted observation)
    y = z - z_pred
    y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # Normalize bearing

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Update state estimate
    state = state + K @ y

    # Normalize theta
    state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))

    # Update covariance estimate
    P = (np.eye(len(state)) - K @ H) @ P

    return state, P


# Update the occupancy grid based on observations
def update_map(occupancy_grid, state, observations, grid_resolution):
    """
    Update the occupancy grid map based on sensor observations.
    
    Args:
        occupancy_grid (np.array): Current occupancy grid.
        state (np.array): Current robot state [x, y, theta].
        observations (list): List of observed features (landmarks) with ranges and bearings.
        grid_resolution (float): Grid resolution in meters.

    Returns:
        np.array: Updated occupancy grid.
    """
    grid_size = occupancy_grid.shape[0]

    for obs in observations:
        r, phi = obs  # Range and bearing

        # Calculate the global position of the observed feature
        x_obs = state[0] + r * np.cos(state[2] + phi)
        y_obs = state[1] + r * np.sin(state[2] + phi)

        # Convert to grid coordinates
        grid_x = int(x_obs / grid_resolution)
        grid_y = int(y_obs / grid_resolution)

        # Ensure the grid coordinates are within bounds
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            occupancy_grid[grid_y, grid_x] = 1  # Mark as occupied

    return occupancy_grid


# Main SLAM loop
def run_slam(control_inputs, observations_list, dt=0.1):
    """
    Run the SLAM algorithm, updating both the robot state and map over time.
    
    Args:
        control_inputs (list): List of control inputs [v, omega] at each time step.
        observations_list (list): List of observations (list of landmarks) at each time step.
        dt (float): Time step between measurements.

    Returns:
        np.array: Final occupancy grid map.
        np.array: Final robot state [x, y, theta].
    """
    # Initialize SLAM components
    occupancy_grid, state, P = initialize_slam()

    # Define noise covariance matrices (adjust these based on system)
    Q = np.diag([0.1, 0.1, np.deg2rad(1)])  # Process noise covariance
    R = np.diag([0.1, np.deg2rad(5)])       # Measurement noise covariance

    for t, (control_input, observations) in enumerate(zip(control_inputs, observations_list)):
        # Predict the next state using the motion model
        state = predict_state(state, control_input, dt)
        P = predict_covariance(P, control_input, Q, dt)

        # For each observation, update the map and the state estimate
        for landmark in observations:
            z_pred = observation_model(state, landmark)
            H = np.array([[1, 0, 0], [0, 1, 0]])  # Simplified Jacobian for this example
            state, P = update_state(state, P, landmark, z_pred, H, R)

        # Update the occupancy grid with the new observations
        occupancy_grid = update_map(occupancy_grid, state, observations, 1.0)

    return occupancy_grid, state


if __name__ == "__main__":
    # Example control inputs and sensor observations for testing
    control_inputs = [[1.0, 0.1] for _ in range(100)]  # Move forward and turn slowly
    observations_list = [[[5.0, 0.5]] for _ in range(100)]  # Simulated range and bearing to landmarks

    # Run the SLAM algorithm
    map, final_state = run_slam(control_inputs, observations_list)

    print("Final robot state:", final_state)
    print("Map after SLAM:\n", map)
