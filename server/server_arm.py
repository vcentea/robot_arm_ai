#!/usr/bin/env python3
import socket
import struct
import json
import time
import sys
from Arm_Lib import Arm_Device

# Port to listen on
PORT = 5000

def main():
    print("Initializing Robot Arm...")
    
    # 1) Initialize the arm and reset the STM32
    try:
        arm = Arm_Device()
        arm.Arm_reset()            # reboot the board
        print("STM32 reset. Waiting for reboot...")
        time.sleep(0.5)            # let the MCU reboot
        
        # 2) Unlock the servo bus
        arm.Arm_Button_Mode(0)
        print("Servo bus unlocked.")
        time.sleep(0.5)            # give the unlock pulse time to settle
        
        # 3) Move to a known "home" pose
        home_pose = [90, 135, 20, 25, 90, 100]  # Relatively safe home position
        print(f"Moving to home position: {home_pose}")
        arm.Arm_serial_servo_write6_array(home_pose, 1500)
        time.sleep(1.5)            # let the arm finish moving
        print("Arm ready.")
    except Exception as e:
        print(f"Error initializing arm: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 4) Start the TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', PORT))
        server.listen()
        print(f"Listening on TCP port {PORT}...")
        
        try:
            while True:
                conn, addr = server.accept()
                print(f"Connection from {addr}")
                with conn:
                    try:
                        # Read message length (4 bytes)
                        length_data = conn.recv(4)
                        if not length_data or len(length_data) != 4:
                            print("Invalid or empty header received")
                            continue
                            
                        length = struct.unpack("!I", length_data)[0]
                        
                        # Read message data
                        data = conn.recv(length).decode('utf-8')
                        if not data:
                            print("Empty message received")
                            continue
                            
                        # Parse the message
                        try:
                            msg = json.loads(data)
                            print(f"Received command: {msg}")
                            
                            if msg["op"] == "move6":
                                # Move all 6 servos
                                arm.Arm_serial_servo_write6_array(msg["pos"], msg.get("time", 1500))
                                response = {"status": "success"}
                                
                            elif msg["op"] == "move1":
                                # Move a single servo
                                servo_id = msg["id"]
                                angle = msg["angle"]
                                time_ms = msg.get("time", 1000)
                                
                                arm.Arm_serial_servo_write(servo_id, angle, time_ms)
                                response = {"status": "success"}
                                
                            elif msg["op"] == "get_angle":
                                # Get the current angle of a servo
                                servo_id = msg["id"]
                                angle = arm.Arm_serial_servo_read(servo_id)
                                response = {"status": "success", "angle": angle}
                                
                            else:
                                response = {"status": "error", "message": f"Unknown operation: {msg['op']}"}
                                
                            # Send response back
                            response_json = json.dumps(response).encode('utf-8')
                            response_header = struct.pack("!I", len(response_json))
                            conn.sendall(response_header + response_json)
                            print(f"Sent response: {response}")
                            
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {data}")
                        except KeyError as e:
                            print(f"Missing required key in command: {e}")
                        except Exception as e:
                            print(f"Error processing command: {e}")
                            
                    except Exception as e:
                        print(f"Connection error: {e}")
                        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            # Clean shutdown of the arm
            try:
                arm.Arm_serial_servo_write6_array(home_pose, 1500)  # Return to home position
                time.sleep(1.5)
                arm.Arm_reset()
                print("Arm reset to safe position.")
            except:
                pass

if __name__ == "__main__":
    main()

