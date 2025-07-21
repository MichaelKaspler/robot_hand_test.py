import serial
import time

print("Arduino Communication Test - Testing Face Control Commands")

try:
    arduino = serial.Serial('COM3', 9600, timeout=2)
    time.sleep(3)
    print("Connected to Arduino on COM3")
    
    commands = ['t', 'c', 'o']
    descriptions = ['thumbs up', 'close hand', 'open hand']
    
    for cmd, desc in zip(commands, descriptions):
        print(f"\nSending: '{cmd}' ({desc})")
        arduino.write(cmd.encode())
        time.sleep(2)
        print("Did the hand move?")
    
    arduino.close()
    print("\nTest completed")

except Exception as e:
    print(f"Error: {e}") 