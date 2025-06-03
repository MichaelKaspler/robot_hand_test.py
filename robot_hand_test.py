import serial
import time

# Try different COM ports to find Arduino
possible_ports = ['COM4', 'COM5', 'COM3', 'COM6', 'COM7', 'COM8']
arduino = None

for port in possible_ports:
    try:
        print(f"Trying to connect to {port}...")
        arduino = serial.Serial(port, 9600)
        print(f"Successfully connected to Arduino on {port}")
        break
    except serial.SerialException:
        print(f"Could not connect to {port}")
        continue

if arduino is None:
    print("No Arduino found. Running in simulation mode...")
    
time.sleep(2)  # Give Arduino time to reset

def send_gesture(command):
    if arduino:
        arduino.write(command.encode())
        print(f"Sent command to Arduino: {command}")
    else:
        print(f"SIMULATION - Would send command: {command}")

# Test: open, close, thumbs up
print("\nTesting robot hand gestures:")
send_gesture('o')  # Open hand
time.sleep(2)

send_gesture('c')  # Close hand
time.sleep(2)

send_gesture('t')  # Thumbs up

if arduino:
    arduino.close()
    print("\nArduino connection closed.")
else:
    print("\nSimulation complete. Connect Arduino to a COM port for real control.")
