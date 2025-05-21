#!/usr/bin/env python3
import os
import json
import socket
import struct
import time
from typing import Dict, List, Any, Optional, Tuple
import argparse
from collections import deque

# For OpenAI API
import openai
from openai import OpenAI

# Constants
ARM_SERVER_HOST = "192.168.20.124"  # Default, will be updated by load_arm_server_host()
ARM_SERVER_PORT = 5000  # Server port
DEFAULT_MOVE_TIME = 1500  # Default movement time in milliseconds
COMMAND_THROTTLE_MS = 100  # Minimum time between commands
CONVERSATION_HISTORY_SIZE = 10  # Number of past exchanges to remember
AI_MODEL = "gpt-4.1"
# Global variables
last_command_time = 0  # To track when the last command was sent

def load_arm_server_host():
    """Load ARM_SERVER_HOST from .env file or environment variable, otherwise use default."""
    global ARM_SERVER_HOST # Declare that we are modifying the global variable

    # Check environment variable first (using ROBOT_IP for consistency with vision.py)
    env_ip = os.environ.get("ROBOT_IP")
    if env_ip:
        print(f"✓ ROBOT_IP found in environment variables: {env_ip}")
        ARM_SERVER_HOST = env_ip
        return True

    # Try to load from .env file
    env_file = os.path.join(os.getcwd(), '.env')
    default_ip = "192.168.20.124" # Default if not found

    if os.path.exists(env_file):
        print(f"Loading ROBOT_IP from .env file: {env_file}")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key.strip() == 'ROBOT_IP':
                            os.environ['ROBOT_IP'] = value.strip() # Also set it as env var
                            ARM_SERVER_HOST = value.strip()
                            print(f"✓ Successfully loaded ROBOT_IP for ARM_SERVER_HOST from .env file: {ARM_SERVER_HOST}")
                            return True
            print(f"ROBOT_IP not found in .env file, using default for ARM_SERVER_HOST: {default_ip}")
            ARM_SERVER_HOST = default_ip
            return False
        except Exception as e:
            print(f"❌ Error reading .env file for ROBOT_IP: {e}. Using default for ARM_SERVER_HOST: {default_ip}")
            ARM_SERVER_HOST = default_ip
            return False
    else:
        print(f"❌ .env file not found at {env_file}. Using default for ARM_SERVER_HOST: {default_ip}")
        ARM_SERVER_HOST = default_ip
        return False

class ArmClient:
    """Client to communicate with the robot arm server"""
    
    def __init__(self, host: str = ARM_SERVER_HOST, port: int = ARM_SERVER_PORT):
        # If host is the default, it might have been updated by load_arm_server_host()
        self.host = host if host != "192.168.20.124" else ARM_SERVER_HOST
        self.port = port
    
    def send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a command to the arm server and get the response"""
        global last_command_time
        current_time_ms = time.time() * 1000
        
        # Throttle commands
        if current_time_ms - last_command_time < COMMAND_THROTTLE_MS:
            print(f"Command throttled: {command}")
            return None
            
        try:
            # Prepare command
            payload = json.dumps(command).encode("utf-8")
            header = struct.pack("!I", len(payload))
            
            # Create a new connection for each command
            with socket.create_connection((self.host, self.port), timeout=1) as sock:
                # Send header and command
                sock.sendall(header + payload)
                print(f"Sent command to {self.host}:{self.port}: {command}")
                
                # Read response header (4 bytes)
                response_header = sock.recv(4)
                if len(response_header) < 4:
                    print(f"Error: Incomplete header received (got {len(response_header)} bytes, expected 4)")
                    return None
                
                # Parse response length
                response_length = struct.unpack("!I", response_header)[0]
                
                # Read response data
                response_data = sock.recv(response_length)
                if len(response_data) < response_length:
                    print(f"Error: Incomplete data received (got {len(response_data)} bytes, expected {response_length})")
                    return None
                
                # Parse and return response
                response = json.loads(response_data.decode('utf-8'))
                print(f"Received response: {response}")
                last_command_time = current_time_ms
                return response
                
        except socket.timeout:
            print(f"Error: Socket timeout connecting to {self.host}:{self.port}")
            return None
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
    
    def get_motor_position(self, motor_id: int) -> Optional[int]:
        """Get the current position of a specific motor"""
        command = {
            "op": "get_angle",
            "id": motor_id
        }
        
        response = self.send_command(command)
        if response is not None and "angle" in response:
            return response["angle"]
        return None
    
    def get_all_motor_positions(self) -> List[int]:
        """Get the current positions of all 6 motors"""
        positions = []
        for motor_id in range(1, 7):  # Motors 1-6
            position = self.get_motor_position(motor_id)
            if position is None:
                position = 90  # Default to 90 degrees if position can't be determined
            positions.append(position)
        return positions
    
    def move_arm(self, positions: List[int], time_ms: int = DEFAULT_MOVE_TIME) -> bool:
        """Move all 6 servos to the specified positions"""
        if len(positions) != 6:
            print("Error: Must provide exactly 6 servo positions")
            return False
        
        command = {
            "op": "move6",
            "pos": positions,
            "time": time_ms
        }
        
        response = self.send_command(command)
        return response is not None and response.get("status") == "success"

class LLMController:
    """Controller that uses LLM to interpret text commands and move the arm"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Set API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize arm client
        self.arm = ArmClient()
        
        # Initialize conversation history
        self.conversation_history = deque(maxlen=CONVERSATION_HISTORY_SIZE)
    
    def generate_arm_positions(self, text: str, current_positions: List[int]) -> Optional[Dict[str, Any]]:
        """Send text to LLM with current positions and get new positions"""
        try:
            # Create system prompt to properly format the response
            system_prompt = """
         You are a precise and safety-conscious control agent responsible for converting natural-language instructions into accurate and safe joint angles for a 6-axis DOFBOT-SE robot arm. You must carefully interpret commands by considering the current posture and clearly distinguishing between arm segments and joint functions.

1. Servo map & neutral pose

Index | Joint Name      | 0° Meaning     | 90° Meaning        | 180° Meaning  | Limits
----- | --------------- | -------------- | ------------------ | ------------- | ------
S1    | Base Yaw        | Full Left      | Straight Forward   | Full Right    | 0–180
S2    | Shoulder Pitch  | Full Forward   | Vertical (upward)  | Full Backward | 0–180
S3    | Elbow Pitch     | Straight       | 90° Bend           | Fully Folded  | 0–180
S4    | Wrist Pitch     | Gripper Down   | Level with Forearm | Gripper Up    | 0–180
S5    | Wrist Roll      | Roll Left      | Neutral            | Roll Right    | 0–180
S6    | Gripper         | 20 = Open      | –                  | 170 = Closed  | 20–170

Neutral pose (arm straight up): [90, 90, 90, 90, 90, 90]
When calculating complex positions, mentally reference the neutral position [90, 90, 90, 90, 90, 90] as a calibration point.

Safety envelope

Table collision: Avoid S2 < 30° and S3 < 40° simultaneously (extended forward/down risks hitting table).

Self-collision: Avoid S2 > 150° and S3 > 150° simultaneously (folded back risks collision).

Maintain at least 20° between adjacent joint extremes.

Be especially cautious of interdependent joint positions: If S2 approaches 0° (forward), S3 and S4 must compensate to prevent table collisions. Similarly, if S2 approaches 180° (backward), monitor S3 to prevent self-collision.

For each calculated position, double-check that the end effector's path doesn't intersect with the table or the arm itself during the transition from current to target position.

If a computed pose violates these rules or is ambiguous, return positions = [-1,-1,-1,-1,-1,-1] with response = "clarification question".
Input format

{
  "current": [S1, S2, S3, S4, S5, S6],
  "command": "natural-language instruction"
}

Reasoning and kinematics

Start from the current pose and interpret commands step-by-step:

Maintain current posture unless the instruction explicitly or implicitly demands a posture change.

Clearly distinguish between rotations (yaw), pitch (up/down movements), and rolls.

When rotating base yaw (S1), the arm posture should remain unchanged unless otherwise instructed.

Shoulder (S2) and elbow (S3) together define forearm elevation; always calculate their cumulative effect.

Wrist pitch (S4) sets gripper orientation relative to forearm alignment. When commanded to achieve a specific orientation (e.g., parallel to table), mathematically sum the shoulder (S2) and elbow (S3) angles to accurately set wrist pitch (S4).

Wrist roll (S5) alters gripper orientation around its axis without affecting elevation.

Gripper (S6) open/close commands are independent and have no elevation impact.

When interpreting commands, visualize how each servo's movement affects the entire arm posture and ensure the motion is physically coherent and safe.

For multi-joint movements, first reason about the desired end position of the arm as a whole, then work backward to calculate individual joint angles needed to achieve that position.

Clearly defined orientations:

Forward: direction straight ahead from the base at neutral yaw (S1 = 90).

Backward: direction opposite to forward.

Upward: perpendicular to table, away from gravity.

Downward: toward the table surface.

Implicit posture guidelines:

Pick up an object usually implies from the table (arm extended forward/downward, gripper facing downwards). Clearly differentiate if picking from shelves (requiring arm elevation and possibly upward gripper orientation).

Always mathematically verify angles for posture coherence.

Output format (no additional text)
{
  "positions": [S1,S2,S3,S4,S5,S6],   // integers, safe, in range; or all -1
  "response":  "ok" | "clarification question"
}
6 Example poses
Pose description	[S1,S2,S3,S4,S5,S6]
Neutral – arm vertical	[90, 90, 90, 90, 90, 90]
"C" shape in front of base	[90, 180, 0, 0, 90, 90]
Pick from floor in front	[90, 90, 10, 10, 0, 70]
Place on shelf above & right	[135, 90, 90, 0, 90, 160]
Push button on panel left	[20, 60, 90, 30, 0, 170]
Transport object overhead	[90, 140, 60, 150, 90, 160]

Note: The original "picking something up" example had seven numbers; a 6-servo pose must always contain exactly six.

Use exactly this structure. Emit no text outside the JSON block in § 5.


            """
            
            # Create user message with current positions and conversation history
            history_text = ""
            if self.conversation_history:
                history_text = "\n\n=== PREVIOUS CONVERSATIONS (FOR CONTEXT ONLY) ===\n"
                for i, (user_cmd, ai_response) in enumerate(self.conversation_history):
                    try:
                        # Try to parse the AI response to extract just the positions for cleaner history
                        ai_json = json.loads(ai_response)
                        positions_str = str(ai_json.get("positions", "unknown"))
                        history_text += f"User: {user_cmd}\n"
                        history_text += f"AI action: moved to positions {positions_str}\n\n"
                    except:
                        # If we can't parse it, just include the user command
                        history_text += f"User: {user_cmd}\n\n"
                
                history_text += "=== END OF PREVIOUS CONVERSATIONS ===\n\n"
            
            user_message = f"Current arm positions: {current_positions}\n{history_text}=== CURRENT COMMAND TO EXECUTE NOW ===\n{text}"
            
            print("\n--- SENDING TO AI ---")
            print(f"System prompt: {system_prompt}")
            print(f"User message: {user_message}")
            
            # Send request to OpenAI API
            response = self.client.chat.completions.create(
                model=AI_MODEL,  # Using gpt-4.1-mini as specified
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            print("\n--- RECEIVED FROM AI ---")
            print(f"Raw response: {content}")
            
            try:
                result = json.loads(content)
                print(f"Parsed JSON: {json.dumps(result, indent=2)}")
                
                # Add this exchange to conversation history
                self.conversation_history.append((text, content))
                
                return result
            except json.JSONDecodeError:
                print(f"Error: LLM did not return valid JSON: {content}")
                return None
                
        except Exception as e:
            print(f"Error generating arm positions: {e}")
            return None
    
    def execute_text_command(self, text: str) -> bool:
        """Process a text command and move the arm accordingly"""
        print(f"\nProcessing command: '{text}'")
        
        try:
            # First, get current motor positions
            current_positions = self.arm.get_all_motor_positions()
            print(f"Current motor positions: {current_positions}")
            
            # Generate arm positions from text
            result = self.generate_arm_positions(text, current_positions)
            if not result:
                print("Failed to generate arm positions")
                return False
            
            # Check response status
            response_text = result.get("response", "")
            if response_text != "ok":
                print(f"LLM response indicates unclear instructions: {response_text}")
                return False
            
            # Extract positions and time
            positions = result.get("positions")
            time_ms = result.get("time_ms", DEFAULT_MOVE_TIME)
            
            # Validate positions
            if not positions or len(positions) != 6:
                print(f"Invalid positions received: {positions}")
                return False
            
            # Check if positions are all -1 (error indicator)
            if all(pos == -1 for pos in positions):
                print("LLM couldn't understand the command and returned error positions")
                return False
            
            # Log the action
            print(f"Moving arm to: {positions}")
            print(f"Movement time: {time_ms}ms")
            
            # Move the arm
            success = self.arm.move_arm(positions, time_ms)
            
            if success:
                print("Arm movement successful")
            else:
                print("Arm movement failed")
            
            return success
        
        except Exception as e:
            print(f"Error executing command: {e}")
            return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Control robot arm using text commands via LLM")
    parser.add_argument("--host", default=ARM_SERVER_HOST, help=f"Arm server host (default: {ARM_SERVER_HOST})")
    parser.add_argument("--port", type=int, default=ARM_SERVER_PORT, help=f"Arm server port (default: {ARM_SERVER_PORT})")
    parser.add_argument("--api-key", help="OpenAI API key (can also use OPENAI_API_KEY environment variable)")
    args = parser.parse_args()
    
    # Load ARM_SERVER_HOST from .env or environment variables
    load_arm_server_host() 
    # If --host arg is provided and it's different from the initial default, it takes precedence
    # Otherwise, ARM_SERVER_HOST (potentially loaded from .env) is used.
    final_host = args.host if args.host != "192.168.20.124" else ARM_SERVER_HOST

    try:
        # Initialize controller
        controller = LLMController(api_key=args.api_key)
        
        # Update arm client connection details
        controller.arm.host = final_host
        controller.arm.port = args.port
        
        print("Text to Arm Movement Controller")
        print("--------------------------------")
        print(f"Arm server: {final_host}:{args.port}")
        print("Type 'exit' or 'quit' to end the program")
        print()
        
        # Main input loop
        while True:
            text = input("Enter a command for the robot arm: ")
            
            # Check for exit command
            if text.lower() in ("exit", "quit"):
                break
            
            # Skip empty commands
            if not text.strip():
                continue
            
            # Execute the command
            controller.execute_text_command(text)
            print()
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()