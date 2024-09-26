import numpy as np

# Path Integration: Dead Reckoning using Odometry and IMU data
def path_integration_update(state, odometry_data, imu_data, dt):
    """
    Update the robot's position using odometry and IMU data (dead reckoning).
    
    Args:
        state (np.array): Current state [x, y, theta].
        odometry_data (np.array): Odometry data [v] (linear velocity).
        imu_data (np.array): IMU data [omega] (angular velocity).
        dt (float): Time step for the update.

    Returns:
        np.array: Updated robot state [x, y, theta].
    """
    # Extract the current position and orientation
    x, y, theta = state

    # Extract velocity from odometry and angular velocity from IMU
    v = odometry_data[0]  # Linear velocity
    omega = imu_data[0]    # Angular velocity

    # Update the state using dead reckoning (path integration)
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    # Normalize the orientation to keep it between -pi and pi
    theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))

    return np.array([x_next, y_next, theta_next])


# Odometry and IMU Data Simulation for Testing
def simulate_odometry_imu_data(time_steps):
    """
    Simulate odometry and IMU data for a simple forward-moving robot.
    
    Args:
        time_steps (int): Number of time steps for simulation.

    Returns:
        list: List of odometry data [v] for each time step.
        list: List of IMU data [omega] for each time step.
    """
    # Simulate constant linear velocity and a slight angular velocity
    odometry_data = [[1.0] for _ in range(time_steps)]  # Constant linear velocity of 1 m/s
    imu_data = [[0.1] for _ in range(time_steps)]       # Constant angular velocity of 0.1 rad/s
    
    return odometry_data, imu_data


# Main Function to Run Path Integration
def run_path_integration(initial_state, odometry_data, imu_data, dt=0.1):
    """
    Run the path integration algorithm using odometry and IMU data.
    
    Args:
        initial_state (np.array): Initial robot state [x, y, theta].
        odometry_data (list): List of odometry data [v] for each time step.
        imu_data (list): List of IMU data [omega] for each time step.
        dt (float): Time step between measurements.

    Returns:
        list: List of robot states [x, y, theta] over time.
    """
    state = initial_state
    states_over_time = [state]

    # Iterate over the data and update the robot's position using dead reckoning
    for i in range(len(odometry_data)):
        state = path_integration_update(state, odometry_data[i], imu_data[i], dt)
        states_over_time.append(state)
    
    return states_over_time


if __name__ == "__main__":
    # Initialize the robot's initial state [x, y, theta]
    initial_state = np.array([0.0, 0.0, 0.0])  # Start at the origin facing east (0 radians)

    # Simulate odometry and IMU data for 100 time steps
    time_steps = 100
    odometry_data, imu_data = simulate_odometry_imu_data(time_steps)

    # Run the path integration algorithm
    states_over_time = run_path_integration(initial_state, odometry_data, imu_data, dt=0.1)

    # Output the final state after path integration
    final_state = states_over_time[-1]
    print("Final state after path integration (x, y, theta):", final_state)

    # Optionally: Print all states over time
    for i, state in enumerate(states_over_time):
        print(f"Time step {i}: State [x, y, theta] = {state}")
