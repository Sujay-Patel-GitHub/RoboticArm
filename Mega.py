import serial
import numpy as np
import time

def calculate_inverse_kinematics(x, y, z):
    # Arm parameters
    L1 = 210  # Upper arm length
    L2 = 222  # Forearm length
    L3 = 16   # Shoulder offset
    L4 = 34   # End-effector offset
    h = 183   # Base height

    # Calculate M1
    R = np.sqrt(x**2 + y**2)
    if R < 1e-6:
        M1 = 0  # Avoid division by zero
    else:
        M1 = np.arctan2(y, x) * 180.0 / np.pi
    if M1 < -179 or M1 > 179:
        return None  # Out of range for M1

    # Adjust for wrist position
    x_w = R
    z_w = z - h

    # Numerical search for M2 and M3
    best_M2, best_M3 = 0, 0
    min_error = float('inf')
    solution_found = False

    for M2 in np.arange(-30, 90.1, 1.0):  # Step size of 1 degree
        theta2 = M2 * np.pi / 180.0
        for M3 in np.arange(0, 180.1, 1.0):
            theta3 = M3 * np.pi / 180.0
            # Compute forward kinematics for wrist position
            x_calc = L1 * np.sin(theta2) + L2 * np.sin(theta3)
            z_calc = L1 * np.cos(theta2) + L3 * np.sin(theta2) + L2 * np.cos(theta2 + theta3) + L4 * np.cos(theta2 - theta3)
            error = np.sqrt((x_calc - x_w)**2 + (z_calc - z_w)**2)
            if error < min_error:
                min_error = error
                best_M2 = M2
                best_M3 = M3
                solution_found = True

    if not solution_found or min_error > 10:  # Tolerance of 10 mm
        return None

    return [M1, best_M2, best_M3]

def main():
    # Configure serial port (replace 'COM3' with your Arduino's COM port)
    try:
        ser = serial.Serial('COM9', 115200, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print("Connected to Arduino on COM3")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    try:
        while True:
            # Get user input for X, Y, Z
            user_input = input("Enter X Y Z coordinates (e.g., '100 200 300') or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

            try:
                x, y, z = map(float, user_input.split())
                angles = calculate_inverse_kinematics(x, y, z)
                if angles is None:
                    print("No valid solution found or position out of range.")
                else:
                    M1, M2, M3 = angles
                    print(f"Calculated angles: M1={M1:.2f}, M2={M2:.2f}, M3={M3:.2f}")
                    # Send angles to Arduino
                    command = f"M1 {M1:.2f} M2 {M2:.2f} M3 {M3:.2f}\n"
                    ser.write(command.encode())
                    print(f"Sent to Arduino: {command.strip()}")

                    # Read Arduino response
                    time.sleep(0.1)  # Allow time for response
                    while ser.in_waiting > 0:
                        response = ser.readline().decode().strip()
                        print(f"Arduino: {response}")
            except ValueError:
                print("Invalid input format. Use: X Y Z (e.g., '100 200 300')")
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")

    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()