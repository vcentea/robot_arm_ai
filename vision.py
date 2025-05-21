import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Import landmark_pb2
import numpy as np
import math
import json
import socket
import struct
import time
import collections
import threading # For voice listener
import requests
import sys # For stderr output
import os # Make sure os is imported
import io # For BytesIO
import wave # For WAV file creation
import argparse # For command-line arguments
import tempfile

# Attempt to import Azure Speech SDK
AZURE_SDK_AVAILABLE = False
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SDK_AVAILABLE = True
    print("✓ Azure Speech SDK imported successfully.")
except ImportError:
    print("Azure Speech SDK not found. Azure STT will be unavailable. Run: pip install azure-cognitiveservices-speech")

# Import LLMController from text_to_arm for speech-to-arm control
try:
    from text_to_arm import LLMController
    TEXT_TO_ARM_AVAILABLE = True
    print("✓ LLMController imported successfully from text_to_arm.py")
    arm_controller = None  # Will be initialized later if OpenAI API key is available
except ImportError as e:
    print(f"Could not import LLMController from text_to_arm: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Check that text_to_arm.py exists in the same directory as vision.py")
    TEXT_TO_ARM_AVAILABLE = False

# Function to load OpenAI API key from .env file if it's not in environment variables
def load_openai_api_key():
    """Try to load OpenAI API key from .env file if it's not in environment variables"""
    if "OPENAI_API_KEY" in os.environ:
        print("✓ OpenAI API key found in environment variables.")
        return True
    
    # Try to load from .env file
    env_file = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_file):
        print(f"Loading API key from .env file: {env_file}")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key.strip() == 'OPENAI_API_KEY':
                            os.environ['OPENAI_API_KEY'] = value.strip()
                            print("✓ Successfully loaded OpenAI API key from .env file")
                            return True
            print("❌ OpenAI API key not found in .env file")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print(f"❌ .env file not found at {env_file}")
    
    print("\n=== HOW TO SET UP OPENAI API KEY ===")
    print("Option 1: Create a .env file in the same directory with the content:")
    print("OPENAI_API_KEY=your_key_here")
    print("\nOption 2: Set the environment variable in the command line:")
    print("Windows: set OPENAI_API_KEY=your_key_here")
    print("Linux/Mac: export OPENAI_API_KEY=your_key_here")
    print("=======================================")
    
    return False

# Function to load Robot IP from .env file or use default
def load_robot_ip():
    """Load ROBOT_IP from .env file or environment variable, otherwise use default."""
    global ROBOT_IP  # Declare that we are modifying the global variable

    # Check environment variable first
    env_ip = os.environ.get("ROBOT_IP")
    if env_ip:
        print(f"✓ ROBOT_IP found in environment variables: {env_ip}")
        ROBOT_IP = env_ip
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
                            os.environ['ROBOT_IP'] = value.strip() # Also set it as env var for consistency
                            ROBOT_IP = value.strip()
                            print(f"✓ Successfully loaded ROBOT_IP from .env file: {ROBOT_IP}")
                            return True
            print(f"ROBOT_IP not found in .env file, using default: {default_ip}")
            ROBOT_IP = default_ip # Use default if not in .env
            return False # Indicate not found in .env, but default is set
        except Exception as e:
            print(f"❌ Error reading .env file for ROBOT_IP: {e}. Using default: {default_ip}")
            ROBOT_IP = default_ip # Use default on error
            return False
    else:
        print(f"❌ .env file not found at {env_file}. Using default ROBOT_IP: {default_ip}")
        ROBOT_IP = default_ip # Use default if .env not found
        return False

# --- Voice Recognition Imports & Setup ---
try:
    print("Attempting to import sounddevice...")
    import sounddevice as sd
    print("sounddevice imported successfully.")

    print("Attempting to import webrtcvad...")
    import webrtcvad
    print("webrtcvad imported successfully.")

    print("Attempting to import queue...")
    import queue
    print("queue imported successfully.")
    
    VOICE_ENABLED = True
    voice_command_queue = queue.Queue() # To pass commands from voice thread to main thread
    llm_queue = queue.Queue() # Queue for messages to send to LLM
except ImportError as e:
    print(f"ImportError encountered: {e}")
    print("One or more voice recognition libraries not found. Voice commands will be disabled.")
    print("Please ensure the following are installed: pip install sounddevice webrtcvad numpy requests")
    VOICE_ENABLED = False # Ensure it's false if imports fail
except Exception as e:
    print(f"An unexpected error occurred during voice library imports: {e}")
    print("Voice commands will be disabled due to an unexpected error.")
    VOICE_ENABLED = False # Ensure it's false if imports fail

# --- STT Service Configuration ---
STT_SERVICE = "whisper"  # Default STT service ("whisper" or "azure")

# --- Azure Speech-to-Text Configuration (used if STT_SERVICE is 'azure') ---
# IMPORTANT: For production, use environment variables or a secure config for API keys.
AZURE_SPEECH_KEY = "DJiwf5zw5k5OnNQ2X3eqPKiysbavf71a5EA7qUGoxxP474ePmXY1JQQJ99BEACYeBjFXJ3w3AAAYACOGyR7Z"
AZURE_SPEECH_REGION = "eastus"
AZURE_SPEECH_LANGUAGE = "en-US"
# Azure SDK objects, will be initialized in voice_listener_thread_func if Azure is used
azure_speech_recognizer = None
azure_push_stream = None

if VOICE_ENABLED: # This block will now run if imports succeed
    # Whisper server configuration (used if STT_SERVICE is 'whisper')
    WHISPER_SERVER_URL = "https://whisper.ainnovate.tech"
    
    # Available models in increasing order of accuracy and size
    AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
    MODEL_NAME = "large-v3" # Default model to request from server
    
    VAD_SENSITIVITY = 3  # 0-3 (strict to lenient)
    VAD_FRAME_MS = 30    # VAD works in 10, 20, or 30 ms frames
    VAD_SILENCE_MS = 600 # How long a pause ends an utterance
    VAD_END_PADDING_MS = 300 # Add extra padding after speech ends to avoid cutting words
    LLM_SILENCE_MS = 3000 # Wait 3 seconds of silence before sending to LLM (reduced from 4000)
    AUDIO_SAMPLERATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_DTYPE = "int16"
    AUDIO_BLOCKSIZE_FRAMES = int(AUDIO_SAMPLERATE * VAD_FRAME_MS / 1000)

    # Add these new noise threshold constants
    DEFAULT_NOISE_ENERGY_THRESHOLD = 500  # Default threshold for noisy environments
    DEFAULT_MIN_VOICE_SNR = 4.0  # Minimum SNR to consider audio as voice (higher = stricter)
    NOISE_ENERGY_THRESHOLD = DEFAULT_NOISE_ENERGY_THRESHOLD  # Will be updated by command line args
    MIN_VOICE_SNR = DEFAULT_MIN_VOICE_SNR  # Will be updated by command line args

    # Azure audio format
    AZURE_AUDIO_FORMAT = speechsdk.audio.AudioStreamFormat(samples_per_second=AUDIO_SAMPLERATE, 
                                                          bits_per_sample=16, 
                                                          channels=AUDIO_CHANNELS) if AZURE_SDK_AVAILABLE else None

    # Debugging flags
    FORCE_RECORD_MODE = False  # Set to False to disable forced recording
    FORCE_RECORD_INTERVAL_SEC = 3.0  # Force a recording every X seconds
    SAVE_DEBUG_AUDIO = False  # Save audio clips locally for debugging
    DEBUG_AUDIO_PATH = "debug_audio"  # Folder to save debug audio files
    DEBUG_VAD_FRAMES = False  # Set to True to enable VAD frame debug output

    voice_command_queue = queue.Queue() # To pass commands from voice thread to main thread
    last_transcribed_text = ""
    voice_status = "Initializing..."
    accumulated_text = "" # To accumulate text before sending to LLM
    last_speech_end_time = 0 # Track when the last speech segment ended

    # Flag to track if we're waiting for LLM response
    waiting_for_llm_response = False

# --- Constants for Gesture Recognizer ---
MODEL_ASSET_PATH = "gesture_recognizer.task" # Path to your downloaded .task file
GESTURE_HISTORY_SIZE = 5 # For smoothing recognized gestures
MIN_GESTURE_CONFIDENCE = 0.5 # Lowered for more sensitivity to raw detection
GESTURE_HOLD_TIME = 0.2 # Reduced hold time for faster stable recognition

# List of all gestures supported by the pre-trained model (for display)
SUPPORTED_GESTURES = [
    "None", "Closed_Fist", "Open_Palm", "Pointing_Up", 
    "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
]

# Custom gestures (detected from landmarks, not from classifier)
CUSTOM_GESTURES = [
    "Pointing_Left", "Pointing_Right"
]

# Robot control parameters
ROBOT_IP = "192.168.20.124"  # Default, will be overridden by load_robot_ip()
ROBOT_PORT = 5000            # From arm.py
SERVO_OPEN_ANGLE = 30        # Robot's gripper open position
SERVO_CLOSED_ANGLE = 170     # Robot's gripper closed position
GRIPPER_SERVO_ID = 6         # The 6th motor for gripper control
ROTATION_SERVO_ID = 1        # The 1st motor for base rotation control
UPDOWN_SERVO_ID = 2          # The 2nd motor for up-down movement
COMMAND_THROTTLE_MS = 100    # Minimum time between commands (milliseconds)
last_command_time = 0        # To track when the last command was sent

# Rotation control parameters
ROTATION_RATE_DEG_PER_SEC = 10   # 5 degrees per 0.5s = 10 deg/sec
MIN_ROTATION_ANGLE = 0           # Minimum rotation angle
MAX_ROTATION_ANGLE = 180         # Maximum rotation angle
TYPICAL_FRAME_INTERVAL_S = 1.0 / 30.0 # Assumed frame interval for smooth first step
current_rotation_angle = 90      # Start in the middle position
last_rotation_time = 0           # To track the last rotation update timestamp
was_rotating_previously = False  # State for continuous rotation logic

# Up-Down control parameters for Motor 2
UPDOWN_RATE_DEG_PER_SEC = 10     # 5 degrees per 0.5s = 10 deg/sec
MIN_UPDOWN_ANGLE = 0            # Minimum up-down angle (arm raised up)
MAX_UPDOWN_ANGLE = 180           # Maximum up-down angle (arm lowered down)
current_updown_angle = 135       # Initial middle position, will be updated from robot
last_updown_time = 0             # To track the last up-down update timestamp
was_updown_previously = False    # State for continuous up-down motion
last_position_query_time = 0     # Track when we last queried the robot for position

# Gripper state variable
current_gripper_angle = (SERVO_OPEN_ANGLE + SERVO_CLOSED_ANGLE) // 2

# Global variables to store the latest results from the gesture recognizer
latest_gesture_results = None
gesture_history = collections.deque(maxlen=GESTURE_HISTORY_SIZE)
last_processed_gesture = "None"
last_gesture_time = 0

# Custom gesture detection parameters
POINTING_DIRECTION_THRESHOLD = 0.12  # Reduced threshold for determining pointing direction
POINTING_ANGLE_THRESHOLD = 35        # Angle threshold in degrees for pointing
POINTING_HYSTERESIS = 0.02           # Hysteresis to prevent flickering
DEBUG_GESTURES = True                # Enable debug visualization for gesture detection

# Track previous pointing state for hysteresis
previous_pointing_state = "None"
pointing_stability_counter = 0       # Counter to provide stability

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles 

# --- Gesture Recognizer Setup ---
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

def print_result_and_update_globals(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture_results
    latest_gesture_results = result

try:
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1, 
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=print_result_and_update_globals)
    recognizer = GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Error initializing Gesture Recognizer: {e}")
    print(f"Please ensure '{MODEL_ASSET_PATH}' is in the correct location and is a valid model file.")
    recognizer = None 

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def send_robot_command(cmd):
    global last_command_time
    current_time_ms = time.time() * 1000
    if current_time_ms - last_command_time < COMMAND_THROTTLE_MS:
        # print(f"Command throttled: {cmd}") # Optional: for debugging throttle
        return False
    try:
        payload = json.dumps(cmd).encode("utf-8")
        header = struct.pack("!I", len(payload))
        with socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=1) as sock:
            sock.sendall(header + payload)
        last_command_time = current_time_ms
        # print(f"Command sent: {cmd}") # Optional: for debugging sent commands
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

# Function to detect custom pointing left/right gestures from hand landmarks
def detect_custom_gestures(hand_landmarks, frame_shape=None):
    global previous_pointing_state, pointing_stability_counter
    
    if not hand_landmarks:
        previous_pointing_state = "None"
        pointing_stability_counter = 0
        return "None", {}

    # Get relevant landmarks
    wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
    index_tip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP]
    thumb_tip = hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
    
    # Check for open palm - if all fingers are extended, prioritize Open_Palm detection
    all_fingers_extended = (
        (index_tip.y < index_pip.y) and
        (middle_tip.y < middle_mcp.y) and
        (ring_tip.y < middle_mcp.y) and
        (pinky_tip.y < middle_mcp.y)
    )
    
    # If it looks like an open palm, don't detect pointing
    if all_fingers_extended:
        previous_pointing_state = "None"
        pointing_stability_counter = 0
        return "None", {"is_likely_palm": True}
    
    # Calculate distances to determine if fingers are extended
    # Extended fingers have tips further from wrist than base joints
    index_extended = (index_tip.y < index_pip.y) # Only check vertical extension (works for any angle)
    
    # Check if other fingers are not extended (curled) relative to middle knuckle
    # This is more permissive for different hand angles
    other_fingers_curled = (
        (middle_tip.y > middle_mcp.y - 0.02) and  # Allow slight extension
        (ring_tip.y > middle_mcp.y - 0.02) and    # Allow slight extension
        (pinky_tip.y > middle_mcp.y - 0.02)       # Allow slight extension
    )
    
    # Calculate horizontal direction using both absolute and relative measurements
    pointing_direction_x = index_tip.x - index_mcp.x
    
    # Measure hand size to scale thresholds based on distance from camera
    hand_width = abs(pinky_mcp.x - index_mcp.x)
    scaled_threshold = max(POINTING_DIRECTION_THRESHOLD, hand_width * 0.35)
    
    # Calculate pointing vector (from MCP to tip)
    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y
    
    # Accept more vertical pointing angles by using the vector rather than just x-direction
    # Calculate vector magnitude (length)
    pointing_magnitude = math.sqrt(dx**2 + dy**2)
    
    # Determine pointing direction based on the dominant component
    is_pointing_right = False
    is_pointing_left = False
    
    if pointing_magnitude > 0:
        # Get angle from straight up
        angle_from_vertical = abs(math.degrees(math.atan2(dx, -dy)))  # -dy to make up direction positive
        
        # Directional check - strong enough horizontal component
        if abs(dx) > pointing_magnitude * 0.55:  # 0.55 corresponds to about 33 degrees from horizontal
            if dx > 0:
                is_pointing_right = True
            else:
                is_pointing_left = True
    
    # Apply hysteresis for stability when transitioning
    current_state = "None"
    if is_pointing_right and index_extended:
        current_state = "Pointing_Right"
    elif is_pointing_left and index_extended:
        current_state = "Pointing_Left"
        
    # Apply hysteresis using the stability counter
    if current_state == previous_pointing_state:
        pointing_stability_counter = min(pointing_stability_counter + 1, 5)
    else:
        # If the new state differs, we need stability counter above 2 to change
        if pointing_stability_counter <= 2:
            current_state = previous_pointing_state
        else:
            pointing_stability_counter = 0
    
    previous_pointing_state = current_state
    
    # Debug variables for visualization
    gesture_debug = {
        "pointing_direction_x": pointing_direction_x,
        "index_extended": index_extended,
        "other_fingers_curled": other_fingers_curled,
        "hand_width": hand_width,
        "scaled_threshold": scaled_threshold,
        "angle_from_vertical": angle_from_vertical if 'angle_from_vertical' in locals() else 0,
        "is_pointing_right": is_pointing_right,
        "is_pointing_left": is_pointing_left,
        "is_palm_like": all_fingers_extended
    }
    
    return current_state, gesture_debug

def control_gripper_direct(recognized_gesture_name):
    global current_gripper_angle
    
    target_angle = current_gripper_angle 
    should_send_command = False
    status = "GRIPPER_IDLE"

    if recognized_gesture_name == "Closed_Fist":
        if current_gripper_angle != SERVO_CLOSED_ANGLE:
            target_angle = SERVO_CLOSED_ANGLE
            should_send_command = True
            status = "CLOSING_GRIPPER (Fist)"
    elif recognized_gesture_name == "Open_Palm":
        if current_gripper_angle != SERVO_OPEN_ANGLE:
            target_angle = SERVO_OPEN_ANGLE
            should_send_command = True
            status = "OPENING_GRIPPER (Palm)"
            
    if should_send_command:
        current_gripper_angle = target_angle 
        cmd = {"op": "move1", "id": GRIPPER_SERVO_ID, "angle": target_angle, "time": 500}
        command_sent_successfully = send_robot_command(cmd)
        return target_angle, status, command_sent_successfully
    
    if recognized_gesture_name == "Closed_Fist" and current_gripper_angle == SERVO_CLOSED_ANGLE:
        status = "ALREADY_CLOSED (Fist)"
    elif recognized_gesture_name == "Open_Palm" and current_gripper_angle == SERVO_OPEN_ANGLE:
        status = "ALREADY_OPEN (Palm)"
        
    return current_gripper_angle, status, False

def control_robot_rotation(recognized_gesture_name, custom_gesture_name):
    global current_rotation_angle, last_rotation_time, was_rotating_previously

    current_time = time.time()
    time_delta_for_calculation = TYPICAL_FRAME_INTERVAL_S  # Default for a new rotation segment

    rotation_direction = 0
    is_rotation_gesture_active = False
    
    # Use custom gesture detection for rotation
    if custom_gesture_name == "Pointing_Right":
        rotation_direction = 1
        is_rotation_gesture_active = True
    elif custom_gesture_name == "Pointing_Left":
        rotation_direction = -1
        is_rotation_gesture_active = True

    if is_rotation_gesture_active:
        if was_rotating_previously and last_rotation_time > 0:
            actual_elapsed = current_time - last_rotation_time
            time_delta_for_calculation = min(actual_elapsed, 3.0 * TYPICAL_FRAME_INTERVAL_S) 
            time_delta_for_calculation = max(time_delta_for_calculation, 0.001)
        max_rotation_step = ROTATION_RATE_DEG_PER_SEC * time_delta_for_calculation
        target_angle_candidate = current_rotation_angle + (rotation_direction * max_rotation_step)
        target_angle_clamped = max(MIN_ROTATION_ANGLE, min(MAX_ROTATION_ANGLE, target_angle_candidate))
        angle_changed_significantly = abs(target_angle_clamped - current_rotation_angle) >= 0.1
        can_move_further = not (
            (rotation_direction > 0 and current_rotation_angle >= MAX_ROTATION_ANGLE) or 
            (rotation_direction < 0 and current_rotation_angle <= MIN_ROTATION_ANGLE)
        )
        if angle_changed_significantly and can_move_further:
            current_rotation_angle = target_angle_clamped
            cmd = {"op": "move1", "id": ROTATION_SERVO_ID, "angle": int(current_rotation_angle), "time": 100}
            cmd_sent = send_robot_command(cmd)
            status_detail = "RIGHT" if rotation_direction > 0 else "LEFT"
            was_rotating_previously = True
            last_rotation_time = current_time
            return current_rotation_angle, f"ROTATING_{status_detail}", cmd_sent
        else:
            was_rotating_previously = True 
            last_rotation_time = current_time
            status = "ROT_AT_LIMIT_OR_NO_SIG_CHANGE"
            if not can_move_further: status = "ROT_AT_LIMIT"
            elif not angle_changed_significantly: status = "ROT_STEP_TOO_SMALL"
            return current_rotation_angle, status, False
    else:
        was_rotating_previously = False
        return current_rotation_angle, "ROT_STABLE (No active gesture)", False

# Function to get current position from robot
def query_robot_position(motor_id):
    global last_position_query_time
    current_time_ms = time.time() * 1000
    
    if current_time_ms - last_position_query_time < 500:
        return None  # Too soon to query again
        
    try:
        cmd = {"op": "get_angle", "id": motor_id}
        payload = json.dumps(cmd).encode("utf-8")
        header = struct.pack("!I", len(payload))
        
        with socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=1) as sock:
            sock.sendall(header + payload)
            print(f"Sent 'get_angle' command for motor {motor_id} to {ROBOT_IP}:{ROBOT_PORT}") # Confirm send
            
            # Read response length (4 bytes)
            response_header = sock.recv(4)
            if len(response_header) < 4:
                print(f"Error querying robot position: Incomplete header received (got {len(response_header)} bytes, expected 4). Check robot server.")
                return None
            
            response_length = struct.unpack("!I", response_header)[0]
            
            # Read response data
            response_data = sock.recv(response_length)
            if len(response_data) < response_length:
                print(f"Error querying robot position: Incomplete data received (got {len(response_data)} bytes, expected {response_length}). Check robot server.")
                return None
            
            response = json.loads(response_data.decode('utf-8')) 
            
            last_position_query_time = current_time_ms 
            if "angle" in response:
                return response["angle"]
            else:
                print(f"Error querying robot position: 'angle' not in response: {response}. Check robot server.")
                return None
    except socket.timeout:
        print(f"Error querying robot position: Socket timeout for motor {motor_id}. Check robot server.")
        return None
    except Exception as e:
        print(f"Error querying robot position: {e}. Check robot server.")
        return None

def control_updown_motion(recognized_gesture_name):
    global current_updown_angle, last_updown_time, was_updown_previously
    
    # Try to get the current position from the robot first
    robot_position = query_robot_position(UPDOWN_SERVO_ID)
    if robot_position is not None:
        current_updown_angle = robot_position  # Update with actual robot position
        print(f"Motor 2 current position: {current_updown_angle}°")
    
    current_time = time.time()
    time_delta_for_calculation = TYPICAL_FRAME_INTERVAL_S  # Default for a new up-down segment

    updown_direction = 0
    is_updown_gesture_active = False
    
    # Use Thumb_Up/Down for motor 2 up/down control
    # REVERSED: Thumb_Up now lowers arm (increases angle toward 180°)
    if recognized_gesture_name == "Thumb_Up":
        updown_direction = 1  # Positive direction moves arm down (servo angle increases toward 180°)
        is_updown_gesture_active = True
    # REVERSED: Thumb_Down now raises arm (decreases angle toward 90°)
    elif recognized_gesture_name == "Thumb_Down":
        updown_direction = -1  # Negative direction moves arm up (servo angle decreases toward 90°)
        is_updown_gesture_active = True

    if is_updown_gesture_active:
        if was_updown_previously and last_updown_time > 0:
            actual_elapsed = current_time - last_updown_time
            time_delta_for_calculation = min(actual_elapsed, 3.0 * TYPICAL_FRAME_INTERVAL_S) 
            time_delta_for_calculation = max(time_delta_for_calculation, 0.001)
        max_updown_step = UPDOWN_RATE_DEG_PER_SEC * time_delta_for_calculation
        target_angle_candidate = current_updown_angle + (updown_direction * max_updown_step)
        target_angle_clamped = max(MIN_UPDOWN_ANGLE, min(MAX_UPDOWN_ANGLE, target_angle_candidate))
        angle_changed_significantly = abs(target_angle_clamped - current_updown_angle) >= 0.1
        can_move_further = not (
            (updown_direction > 0 and current_updown_angle >= MAX_UPDOWN_ANGLE) or 
            (updown_direction < 0 and current_updown_angle <= MIN_UPDOWN_ANGLE)
        )
        if angle_changed_significantly and can_move_further:
            current_updown_angle = target_angle_clamped
            cmd = {"op": "move1", "id": UPDOWN_SERVO_ID, "angle": int(current_updown_angle), "time": 100}
            cmd_sent = send_robot_command(cmd)
            # REVERSED: Status message direction also reversed
            status_detail = "UP" if updown_direction < 0 else "DOWN"
            was_updown_previously = True
            last_updown_time = current_time
            return current_updown_angle, f"MOVING_{status_detail}", cmd_sent
        else:
            was_updown_previously = True 
            last_updown_time = current_time
            status = "UPDOWN_AT_LIMIT_OR_NO_SIG_CHANGE"
            if not can_move_further: 
                if updown_direction < 0:
                    status = "AT_UPPER_LIMIT (90°)"
                else:
                    status = "AT_LOWER_LIMIT (180°)"
            elif not angle_changed_significantly: status = "UPDOWN_STEP_TOO_SMALL"
            return current_updown_angle, status, False
    else:
        was_updown_previously = False
        return current_updown_angle, "UPDOWN_STABLE (No active gesture)", False

def process_recognized_gestures(frame_shape):
    global latest_gesture_results, gesture_history, last_processed_gesture, last_gesture_time
    global current_gripper_angle, current_rotation_angle, current_updown_angle

    display_gesture_name = "None"
    final_recognized_gesture = "None"
    custom_gesture_name = "None"
    hand_landmarks_for_drawing = None
    gesture_debug_info = None
    
    actual_gripper_angle_val = current_gripper_angle
    gripper_status_val = "GRIPPER_IDLE"
    gripper_cmd_sent_val = False

    actual_rotation_angle_val = current_rotation_angle
    rotation_status_val = "ROT_IDLE"
    rotation_cmd_sent_val = False
    
    actual_updown_angle_val = current_updown_angle
    updown_status_val = "UPDOWN_IDLE"
    updown_cmd_sent_val = False

    thumb_tip_px_val, index_tip_px_val, dist_px_val = None, None, None

    if latest_gesture_results and latest_gesture_results.gestures:
        current_time = time.time()
        if latest_gesture_results.gestures and latest_gesture_results.gestures[0]:
            top_gesture = latest_gesture_results.gestures[0][0]
            if top_gesture.score > MIN_GESTURE_CONFIDENCE:
                display_gesture_name = top_gesture.category_name
                gesture_history.append(display_gesture_name)

        if len(gesture_history) == GESTURE_HISTORY_SIZE and all(g == gesture_history[0] for g in gesture_history):
            if gesture_history[0] != last_processed_gesture:
                last_processed_gesture = gesture_history[0]
                last_gesture_time = current_time
                final_recognized_gesture = last_processed_gesture
            elif current_time - last_gesture_time < GESTURE_HOLD_TIME: 
                final_recognized_gesture = last_processed_gesture
            else: 
                if display_gesture_name == last_processed_gesture: 
                    final_recognized_gesture = display_gesture_name
                    last_gesture_time = current_time 
                else: 
                    final_recognized_gesture = "None"
                    last_processed_gesture = "None" 

        if latest_gesture_results.hand_landmarks:
            hand_landmarks_for_drawing = latest_gesture_results.hand_landmarks[0]
            
            # Detect custom gestures using landmarks
            custom_gesture_result = detect_custom_gestures(hand_landmarks_for_drawing, frame_shape)
            custom_gesture_name = custom_gesture_result[0]
            gesture_debug_info = custom_gesture_result[1] if len(custom_gesture_result) > 1 else None
            
            # Prioritize standard "Open_Palm" gesture over custom gestures
            if final_recognized_gesture == "Open_Palm":
                custom_gesture_name = "None"  # Don't allow custom gestures to override Open_Palm
            
            # NEW: If Thumb_Up or Thumb_Down is the stable gesture, ignore custom pointing gestures
            if final_recognized_gesture == "Thumb_Up" or final_recognized_gesture == "Thumb_Down":
                custom_gesture_name = "None"
            
            if hand_landmarks_for_drawing and len(hand_landmarks_for_drawing) >= max(mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP) + 1:
                h, w = frame_shape[:2]
                thumb_tip_px_val = (int(hand_landmarks_for_drawing[mp.solutions.hands.HandLandmark.THUMB_TIP].x * w),
                                    int(hand_landmarks_for_drawing[mp.solutions.hands.HandLandmark.THUMB_TIP].y * h))
                index_tip_px_val = (int(hand_landmarks_for_drawing[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                    int(hand_landmarks_for_drawing[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * h))
                dist_px_val = calculate_distance(thumb_tip_px_val, index_tip_px_val)

        # Control Logic for all gestures
        actual_gripper_angle_val, gripper_status_val, gripper_cmd_sent_val = control_gripper_direct(final_recognized_gesture)
        actual_rotation_angle_val, rotation_status_val, rotation_cmd_sent_val = control_robot_rotation(final_recognized_gesture, custom_gesture_name)
        actual_updown_angle_val, updown_status_val, updown_cmd_sent_val = control_updown_motion(final_recognized_gesture)

    return (display_gesture_name, final_recognized_gesture, custom_gesture_name, hand_landmarks_for_drawing, 
            actual_gripper_angle_val, gripper_status_val, gripper_cmd_sent_val,
            actual_rotation_angle_val, rotation_status_val, rotation_cmd_sent_val,
            actual_updown_angle_val, updown_status_val, updown_cmd_sent_val,
            thumb_tip_px_val, index_tip_px_val, dist_px_val, gesture_debug_info)

def visualize_arm(frame, hand_landmarks_list, pose_landmarks=None):
    h, w, _ = frame.shape
    
    if hand_landmarks_list: 
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else 0.0,
                presence=lm.presence if hasattr(lm, 'presence') else 0.0
            ) for lm in hand_landmarks_list
        ])
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS, 
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        wrist = hand_landmarks_list[mp.solutions.hands.HandLandmark.WRIST]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h))
        cv2.circle(frame, wrist_pos, 8, (100, 100, 255), -1)
        
    if pose_landmarks:
        RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value 
        pass 
    
    return frame

def toggle_tracking_mode(mode):
    modes = ["GESTURE", "ARM_GESTURE"] 
    return modes[(modes.index(mode) + 1) % len(modes)]

def draw_gripper_visualization(frame, angle):
    h, w, _ = frame.shape
    viz_x = w - 120
    viz_y = 150 
    viz_width = 100
    viz_height = 60
    
    cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), (100, 100, 100), 1)
    
    if (SERVO_CLOSED_ANGLE - SERVO_OPEN_ANGLE) == 0:
        open_ratio = 0.5
    else:
        open_ratio = (SERVO_CLOSED_ANGLE - angle) / (SERVO_CLOSED_ANGLE - SERVO_OPEN_ANGLE)
    open_ratio = max(0, min(1, open_ratio))
    
    jaw_length = 40
    jaw_thickness = 8
    center_y = viz_y + viz_height // 2
    max_half_gap = 15 
    current_half_gap = int(open_ratio * max_half_gap)

    cv2.rectangle(frame, 
                  (viz_x + viz_width//2 - jaw_thickness//2, center_y - jaw_length//2 - current_half_gap),
                  (viz_x + viz_width//2 + jaw_thickness//2, center_y - current_half_gap),
                  (0, 200, 200), -1)
    cv2.rectangle(frame, 
                  (viz_x + viz_width//2 - jaw_thickness//2, center_y + current_half_gap),
                  (viz_x + viz_width//2 + jaw_thickness//2, center_y + jaw_length//2 + current_half_gap),
                  (0, 200, 200), -1)

    label = "PARTIAL"
    if open_ratio > 0.85: label = "OPEN"
    elif open_ratio < 0.15: label = "CLOSED"
    cv2.putText(frame, label, (viz_x + 5, viz_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    
    return frame

# At the top of the file, add these global variables with the other globals
error_message = None
error_message_time = 0
ERROR_DISPLAY_DURATION = 5  # seconds to display error message

# --- Voice Listener Thread Function ---
def voice_listener_thread_func():
    global last_transcribed_text, voice_status, accumulated_text, last_speech_end_time, llm_queue
    global azure_speech_recognizer, azure_push_stream
    global waiting_for_llm_response

    if not VOICE_ENABLED:
        voice_status = "Disabled (due to import or init error)"
        return

    try:
        print(f"\n=== Initializing Voice Recognition ===")
        print(f"Selected STT Service: {STT_SERVICE.upper()}")

        # --- COMMON INITIALIZATION ---
        # Initialize VAD for both Azure and Whisper
        try:
            vad = webrtcvad.Vad(VAD_SENSITIVITY)
            print("✓ VAD initialized successfully")
            
            # Add these variables for adaptive VAD
            background_energy_samples = []
            adaptive_vad_sensitivity = VAD_SENSITIVITY
            last_vad_adjustment_time = time.monotonic()
            
        except Exception as vad_error:
            print(f"Error initializing VAD: {vad_error}")
            voice_status = f"VAD Error: {vad_error}"
            return

        # --- SERVICE-SPECIFIC INITIALIZATION ---
        if STT_SERVICE == "azure":
            if not AZURE_SDK_AVAILABLE:
                voice_status = "Azure SDK not found. Voice disabled."
                print("❌ Azure STT selected, but SDK not available.")
                return

            print(f"Initializing Azure Speech-to-Text: Region '{AZURE_SPEECH_REGION}', Language '{AZURE_SPEECH_LANGUAGE}'")
            try:
                # Create a speech configuration
                speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
                speech_config.speech_recognition_language = AZURE_SPEECH_LANGUAGE
                print("✓ Azure speech config created")
            except Exception as e:
                print(f"❌ Failed to initialize Azure speech config: {e}")
                voice_status = f"Azure Config Error: {e}"
                return
                
        elif STT_SERVICE == "whisper":
            print(f"Initializing Whisper connection: {WHISPER_SERVER_URL}")
            try:
                # Check server connection
                response = requests.get(f"{WHISPER_SERVER_URL}/health", timeout=5)
                if response.status_code == 200:
                    server_info = response.json()
                    print(f"✓ Connected to Whisper server")
                    print(f"Whisper Server status: {server_info['status']}")
                    voice_status = "Whisper: Ready"
                else:
                    print(f"Whisper server responded with status code: {response.status_code}")
                    voice_status = f"Whisper Server issue: {response.status_code}"
                    return
            except Exception as server_error:
                print(f"Error connecting to Whisper server: {server_error}")
                voice_status = f"Whisper Server Error: {server_error}"
                return

    except Exception as e:
        print(f"Error initializing voice components: {e}")
        voice_status = f"Init Error: {e}"
        return

    # --- COMMON AUDIO PROCESSING SETUP ---
    audio_buffer = queue.Queue()  # Common audio buffer for both services
    speech_frames = bytearray()   # Common buffer to accumulate speech segments
    utterance_started_time = None # Common tracking for both services
    
    # Add buffer for background noise sampling
    background_frames = bytearray()  # Buffer to collect background noise for noise reduction
    is_collecting_background = True   # Start by collecting background noise
    background_collection_end_time = time.monotonic() + 2.0  # Collect 2 seconds of background
    
    # Initialize is_speech to False before the loop
    is_speech = False
    
    def _audio_callback(indata, frames, time_info, status):
        """Common audio callback for both services"""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        # Push audio data to common buffer
        audio_buffer.put(bytes(indata))

    # --- MICROPHONE SETUP ---
    try:
        devices = sd.query_devices()
        default_device_info = sd.query_devices(kind='input')
        default_device_id = default_device_info['index']
        
        print(f"\nUsing microphone: {default_device_info['name']} (device {default_device_id})")
        
        # Mic testing logic
        device_to_use = None
        mic_indices_to_try = [default_device_id] + [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0 and i != default_device_id]
        
        for mic_idx in mic_indices_to_try:
            print(f"Attempting to test microphone: {devices[mic_idx]['name']} (device {mic_idx})")
            if try_different_mic(mic_idx):
                device_to_use = mic_idx
                print(f"✓ Successfully selected microphone: {devices[device_to_use]['name']} (device {device_to_use})")
                break
            else:
                print(f"❌ Microphone test failed for: {devices[mic_idx]['name']} (device {mic_idx})")

        if device_to_use is None:
            print("❌ CRITICAL: No working microphone found after testing. Voice input will not work.")
            voice_status = "No working microphone found."
            return
        
        # --- MAIN AUDIO CAPTURE AND PROCESSING LOOP ---
        with sd.RawInputStream(samplerate=AUDIO_SAMPLERATE, 
                               blocksize=AUDIO_BLOCKSIZE_FRAMES,
                               dtype=AUDIO_DTYPE, 
                               channels=AUDIO_CHANNELS,
                               device=device_to_use,
                               callback=_audio_callback):
            
            print(f"🎙️ sd.RawInputStream started. Using {STT_SERVICE.upper()} for STT. Waiting for voice input...")
            voice_status = f"Listening ({STT_SERVICE})..."
            
            force_collecting = False
            force_frames = bytearray()
            last_force_record_time = time.monotonic()
            frame_counter = 0
            
            # Collect initial background noise sample
            print("Collecting initial background noise sample (2 seconds)...")
            voice_status = "Calibrating noise levels..."
            
            while True:
                current_time_monotonic = time.monotonic()
                
                # --- BACKGROUND NOISE COLLECTION ---
                if is_collecting_background and current_time_monotonic < background_collection_end_time:
                    try:
                        frame = audio_buffer.get(timeout=0.1)
                        background_frames += frame
                        continue  # Skip other processing during initial collection
                    except queue.Empty:
                        pass
                elif is_collecting_background:
                    is_collecting_background = False
                    print(f"Collected {len(background_frames)} bytes of background noise for calibration")
                    voice_status = f"Listening ({STT_SERVICE})..."
                
                # --- COMMON LOGIC: Check if accumulated text should be sent to LLM ---
                if (accumulated_text and 
                    (current_time_monotonic - last_speech_end_time) * 1000 > LLM_SILENCE_MS and 
                    not waiting_for_llm_response):
                    
                    print(f"\n🤖 Sending to LLM ({STT_SERVICE.upper()}): '{accumulated_text}' (silence duration met)")
                    llm_queue.put(accumulated_text)
                    waiting_for_llm_response = True  # Set flag to indicate we're waiting for response
                    voice_status = f"Sent to LLM ({STT_SERVICE}), waiting for response..."
                    accumulated_text = ""
                
                # --- FORCE RECORDING MODE LOGIC (Common) ---
                if FORCE_RECORD_MODE:
                    time_since_last_force = current_time_monotonic - last_force_record_time
                    if not force_collecting and time_since_last_force > FORCE_RECORD_INTERVAL_SEC:
                        force_collecting = True
                        force_frames = bytearray()
                        print(f"⏺️ FORCE MODE: Starting forced recording")
                    
                    if force_collecting and time_since_last_force > (FORCE_RECORD_INTERVAL_SEC + 2.0):
                        force_collecting = False
                        last_force_record_time = current_time_monotonic
                        print(f"⏹️ FORCE MODE: Ending forced recording, processing {len(force_frames)} bytes")
                        
                        # Process forced recording
                        if len(force_frames) > 0:
                            # Transcribe with selected service
                            transcribe_audio_segment(force_frames, is_force_mode=True)
                
                # Get the next audio frame
                try:
                    frame = audio_buffer.get(timeout=0.1)  # Small timeout to prevent CPU spinning
                    
                    # Add to force buffer if in force collection mode
                    if FORCE_RECORD_MODE and force_collecting:
                        force_frames += frame
                    
                    # --- COMMON VAD PROCESSING ---
                    # Process VAD first to determine if the frame contains speech
                    is_speech = vad.is_speech(frame, AUDIO_SAMPLERATE)
                    frame_counter += 1
                    
                    # --- ADAPTIVE VAD PROCESSING ---
                    # Sample background noise and adjust VAD sensitivity periodically
                    pcm_data = np.frombuffer(frame, dtype=AUDIO_DTYPE)
                    frame_energy = np.mean(np.abs(pcm_data))
                    
                    # Update background noise samples during non-speech periods
                    if not is_speech and utterance_started_time is None:
                        if len(background_energy_samples) < 50:  # Keep last 50 samples
                            background_energy_samples.append(frame_energy)
                        else:
                            background_energy_samples.pop(0)
                            background_energy_samples.append(frame_energy)
                    
                    # Adjust VAD sensitivity based on background noise every 5 seconds
                    if current_time_monotonic - last_vad_adjustment_time > 5.0 and len(background_energy_samples) >= 10:
                        avg_background = sum(background_energy_samples) / len(background_energy_samples)
                        prev_sensitivity = adaptive_vad_sensitivity
                        
                        # Increase sensitivity (lower number) in quiet environments
                        # Decrease sensitivity (higher number) in noisy environments
                        if avg_background > 1000:  # Very noisy
                            adaptive_vad_sensitivity = min(3, adaptive_vad_sensitivity + 1)
                        elif avg_background < 200:  # Very quiet
                            adaptive_vad_sensitivity = max(0, adaptive_vad_sensitivity - 1)
                        
                        # Apply the new sensitivity if it changed
                        if adaptive_vad_sensitivity != prev_sensitivity:
                            vad = webrtcvad.Vad(adaptive_vad_sensitivity)
                            print(f"Adjusted VAD sensitivity to {adaptive_vad_sensitivity} (background energy: {avg_background:.1f})")
                        
                        last_vad_adjustment_time = current_time_monotonic
                    
                    # Process speech frames based on VAD result
                    if is_speech:
                        speech_frames += frame
                        if utterance_started_time is None:
                            utterance_started_time = current_time_monotonic
                            voice_status = f"Hearing voice ({STT_SERVICE})..."
                    
                    # Track when speech ended, but include padding time to avoid cutting words
                    elif utterance_started_time:
                        speech_frames += frame  # Always add the current frame during active speech or padding
                        
                        # Check if we're past the silence threshold including padding
                        silence_duration = current_time_monotonic - utterance_started_time
                        required_silence = (VAD_SILENCE_MS + VAD_END_PADDING_MS) / 1000.0
                        
                        if silence_duration > required_silence:
                            voice_status = f"Processing ({STT_SERVICE})..."
                            last_speech_end_time = current_time_monotonic # Update time when speech ended
                            
                            # Process speech segment if long enough
                            sample_width_bytes = np.dtype(AUDIO_DTYPE).itemsize
                            if len(speech_frames) > AUDIO_SAMPLERATE / 4 * sample_width_bytes:
                                # Transcribe with selected service
                                transcribe_audio_segment(speech_frames, background_frames, is_force_mode=False)
                            else:
                                print(f"Speech segment too short ({len(speech_frames)} bytes), ignoring")
                            
                            # Reset for next utterance
                            speech_frames = bytearray()
                            utterance_started_time = None
                            voice_status = f"Listening ({STT_SERVICE})..."
                    
                    elif not is_speech and utterance_started_time is None:
                        voice_status = f"Listening ({STT_SERVICE})..."
                
                except queue.Empty:
                    # No audio data available, just continue and check other conditions
                    pass

    except Exception as e:
        print(f"Exception in voice listener thread: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        voice_status = f"Thread Error ({STT_SERVICE}): {e}"

# Function to transcribe audio with the selected STT service
def transcribe_audio_segment(audio_data, background_noise=None, is_force_mode=False):
    """Transcribe an audio segment using the selected STT service with noise filtering"""
    global last_transcribed_text, voice_status, accumulated_text, last_speech_end_time
    
    prefix = "FORCE MODE" if is_force_mode else ""
    
    # Apply noise filtering and enhancement
    try:
        # Calculate original audio quality metrics
        pcm_data = np.frombuffer(audio_data, dtype=AUDIO_DTYPE)
        original_duration = len(pcm_data) / AUDIO_SAMPLERATE
        original_energy = np.mean(np.abs(pcm_data))
        original_max = np.max(np.abs(pcm_data))
        original_snr = original_max / (original_energy + 1e-10)
        
        print(f"🎙️ {prefix} Original audio: {len(audio_data)} bytes, duration={original_duration:.2f}s, energy={original_energy:.1f}, SNR={original_snr:.1f}")
        
        # Early rejection - if the energy is very low or SNR is poor, reject immediately
        if original_energy < NOISE_ENERGY_THRESHOLD * 0.5 or original_snr < MIN_VOICE_SNR * 0.8:
            print(f"🔇 {prefix} Rejected audio at pre-filtering stage: energy={original_energy:.1f}, SNR={original_snr:.1f}")
            # Count this as speech ending for LLM silence timer
            last_speech_end_time = time.monotonic()
            return
        
        # 1. Apply bandpass filter for voice frequencies
        filtered_audio = apply_bandpass_filter(audio_data)
        
        # 2. Apply noise reduction
        denoised_audio = denoise_audio(filtered_audio)
        
        # Calculate enhanced audio metrics
        enhanced_pcm = np.frombuffer(denoised_audio, dtype=AUDIO_DTYPE)
        enhanced_energy = np.mean(np.abs(enhanced_pcm))
        enhanced_max = np.max(np.abs(enhanced_pcm))
        enhanced_snr = enhanced_max / (enhanced_energy + 1e-10)
        
        print(f"🔊 {prefix} Enhanced audio: {len(denoised_audio)} bytes, energy={enhanced_energy:.1f}, SNR={enhanced_snr:.1f}")
        
        # Use the enhanced audio for transcription
        audio_data = denoised_audio
        
    except Exception as e:
        print(f"Error during audio enhancement: {e}")
        # Continue with original audio if enhancement fails
    
    # Minimum thresholds for audio quality - now using constants
    MIN_AUDIO_DURATION = 0.5  # Seconds
    MIN_AUDIO_ENERGY = NOISE_ENERGY_THRESHOLD  # Use the configurable threshold
    MIN_SIGNAL_TO_NOISE = MIN_VOICE_SNR  # Use the configurable SNR threshold
    
    # Re-check audio quality after enhancement
    pcm_data = np.frombuffer(audio_data, dtype=AUDIO_DTYPE)
    duration_seconds = len(pcm_data) / AUDIO_SAMPLERATE
    audio_energy = np.mean(np.abs(pcm_data))
    audio_max = np.max(np.abs(pcm_data))
    signal_to_noise = audio_max / (audio_energy + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Log details but with different verbosity based on if it passes quality check
    quality_check_passed = (
        duration_seconds >= MIN_AUDIO_DURATION and
        audio_energy >= MIN_AUDIO_ENERGY and
        signal_to_noise >= MIN_SIGNAL_TO_NOISE
    )
    
    quality_info = f"duration={duration_seconds:.2f}s, energy={audio_energy:.1f}, SNR={signal_to_noise:.1f}"
    
    if quality_check_passed:
        print(f"🎙️ {prefix} Final audio clip to transcribe: {len(audio_data)} bytes, {quality_info}")
    else:
        reasons = []
        if duration_seconds < MIN_AUDIO_DURATION:
            reasons.append(f"too short ({duration_seconds:.2f}s < {MIN_AUDIO_DURATION}s)")
        if audio_energy < MIN_AUDIO_ENERGY:
            reasons.append(f"low energy ({audio_energy:.1f} < {MIN_AUDIO_ENERGY})")
        if signal_to_noise < MIN_SIGNAL_TO_NOISE:
            reasons.append(f"poor SNR ({signal_to_noise:.1f} < {MIN_SIGNAL_TO_NOISE})")
        
        print(f"🔇 {prefix} Skipping low-quality audio: {', '.join(reasons)}")
        # Count this as speech ending for LLM silence timer
        last_speech_end_time = time.monotonic()
        return
    
    try:
        if STT_SERVICE == "azure":
            # Prepare WAV for Azure
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(np.dtype(AUDIO_DTYPE).itemsize)
                wf.setframerate(AUDIO_SAMPLERATE)
                wf.writeframes(pcm_data.tobytes())
            wav_bytes.seek(0)
            
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_recognition_language = AZURE_SPEECH_LANGUAGE
            
            # Configure Azure to use an in-memory push stream to avoid disk I/O
            if AZURE_AUDIO_FORMAT is None:
                print("❌ AZURE_AUDIO_FORMAT not initialized. Cannot use Azure STT in-memory stream.")
                return

            push_stream = speechsdk.audio.PushAudioInputStream(stream_format=AZURE_AUDIO_FORMAT)
            push_stream.write(wav_bytes.getvalue()) # wav_bytes contains the full WAV audio
            push_stream.close() # Signal the end of the stream (important!)

            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            print(f"🚀 {prefix} Sending audio to Azure (in-memory)...")
            result = speech_recognizer.recognize_once()
            
            # No temporary file to clean up
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text.strip()
                print(f"👤 AZURE {prefix} User: {text}")
                last_transcribed_text = text
                
                # Add to accumulated text for LLM
                if accumulated_text:
                    accumulated_text += " " + text
                else:
                    accumulated_text = text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print(f"Azure {prefix} no speech recognized.")
                voice_status = "No speech detected"
                # Count this as speech ending for LLM silence timer
                last_speech_end_time = time.monotonic()
            else:
                print(f"Azure {prefix} other result: {result.reason}")
        
        elif STT_SERVICE == "whisper":
            # Prepare WAV for Whisper
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(np.dtype(AUDIO_DTYPE).itemsize)
                wf.setframerate(AUDIO_SAMPLERATE)
                wf.writeframes(pcm_data.tobytes())
            wav_bytes.seek(0)
            
            # Send to server
            files = {'file': ('speech.wav', wav_bytes, 'audio/wav')}
            data = {'model': MODEL_NAME}
            
            print(f"🚀 {prefix} Sending enhanced audio to Whisper server...")
            response = requests.post(
                f"{WHISPER_SERVER_URL}/transcribe",
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["text"].strip()
                
                if text:
                    print(f"👤 WHISPER {prefix} User: {text}")
                    last_transcribed_text = text
                    
                    # Add to accumulated text for LLM
                    if accumulated_text:
                        accumulated_text += " " + text
                    else:
                        accumulated_text = text
                else:
                    print(f"Whisper {prefix} no text in response")
                    # Count this as speech ending for LLM silence timer
                    last_speech_end_time = time.monotonic()
            else:
                print(f"Whisper {prefix} server error: {response.status_code} - {response.text}")
                voice_status = f"Whisper Error: {response.status_code}"
                # Count server errors as speech ending for LLM silence timer
                last_speech_end_time = time.monotonic()
    
    except Exception as e:
        print(f"Error transcribing audio with {STT_SERVICE}: {e}")
        voice_status = f"Transcription Error: {str(e)[:30]}..."
        import traceback
        traceback.print_exc()
        # Count errors as speech ending for LLM silence timer
        last_speech_end_time = time.monotonic()

# LLM processing thread function
def llm_processing_thread_func():
    """Thread function to process messages sent to the LLM"""
    global error_message, error_message_time
    global waiting_for_llm_response
    
    while True:
        try:
            message = llm_queue.get()
            if message and TEXT_TO_ARM_AVAILABLE and arm_controller:
                print(f"\n📝 LLM PROCESSING: '{message}'")
                try:
                    print(f"Sending accumulated text to arm controller: '{message}'")
                    success = arm_controller.execute_text_command(message)
                    
                    if success:
                        print("✅ Arm movement successful")
                        error_message = None # Clear previous errors
                    else:
                        detailed_error = "Failed to execute command or command unclear." # Default error
                        if arm_controller.conversation_history:
                            last_user_cmd, last_ai_resp_str = arm_controller.conversation_history[-1]
                            if last_user_cmd == message: # Ensure it's for the current command
                                try:
                                    last_ai_resp_json = json.loads(last_ai_resp_str)
                                    if last_ai_resp_json.get("response") and last_ai_resp_json.get("response") != "ok":
                                        detailed_error = last_ai_resp_json["response"]
                                    elif not last_ai_resp_json.get("response"): # Malformed response
                                        detailed_error = "LLM response format error."
                                    elif last_ai_resp_json.get("response") == "ok": # LLM was ok, but arm move failed
                                         detailed_error = "Arm movement failed after LLM confirmation."
                                except json.JSONDecodeError:
                                    detailed_error = "Error parsing LLM response."
                        
                        error_message = detailed_error
                        error_message_time = time.time()
                        print(f"❌ LLM/Arm Error: {error_message}")

                except Exception as e: # Catch errors from execute_text_command itself (e.g., network issues)
                    error_message = f"LLM Processing Error: {str(e)}"
                    error_message_time = time.time()
                    print(f"❌ Exception in LLM command execution: {e}")
            elif message: # Message exists but arm_controller not available
                print(f"\n📝 LLM received message but text_to_arm is not available: '{message}'")
                error_message = "Arm controller not available for LLM."
                error_message_time = time.time()
                
            llm_queue.task_done()
            
            # Clear waiting flag so new messages can be sent
            waiting_for_llm_response = False
            print("✓ LLM processing complete. Ready for new commands.")
            
        except Exception as e:
            print(f"Error in LLM processing thread: {e}")
            # Still clear the flag even if an error occurred
            waiting_for_llm_response = False

# Function to change the voice recognition model on the server
def change_voice_model(new_model_name):
    """Change the ASR model on the server"""
    global MODEL_NAME
    
    if not VOICE_ENABLED:
        print("Voice recognition is disabled")
        return False
    
    if new_model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{new_model_name}' not in available models: {AVAILABLE_MODELS}")
        return False
    
    if new_model_name == MODEL_NAME:
        print(f"Already using model '{MODEL_NAME}'")
        return True
    
    try:
        # Change model on the server
        response = requests.post(
            f"{WHISPER_SERVER_URL}/change_model",
            data={"model_name": new_model_name},
            timeout=5
        )
        
        if response.status_code == 200:
            MODEL_NAME = new_model_name
            print(f"Model changed to {MODEL_NAME} on server")
            return True
        else:
            print(f"Error changing model on server: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error changing model: {e}")
        return False

# Try different microphones if needed
def try_different_mic(device_id):
    """Try to use a different microphone device"""
    try:
        devices = sd.query_devices()
        input_devices = []
        
        # Find all input devices
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append((i, dev['name']))
        
        if not input_devices:
            print("No input devices found!")
            return None
            
        # If device_id is out of range, use the first available
        if device_id >= len(input_devices):
            device_id = 0
            
        # Get the selected device
        selected_id, selected_name = input_devices[device_id]
        print(f"\nTrying microphone: {selected_name} (device {selected_id})")
        
        # Test recording with this device
        TEST_DURATION = 0.5  # seconds
        test_frames = sd.rec(
            int(TEST_DURATION * AUDIO_SAMPLERATE),
            samplerate=AUDIO_SAMPLERATE,
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            device=selected_id
        )
        sd.wait()  # Wait for recording to finish
        
        # Check if we got audio by looking at its energy or max amplitude
        if test_frames is not None and np.max(np.abs(test_frames)) > 0: # Check for non-zero signal
            audio_max = np.max(np.abs(test_frames))
            print(f"Test recording for {selected_name} successful. Max amplitude: {audio_max}")
            return True # Indicate success
        else:
            print(f"Failed to record or silent audio from device {selected_name} (device {selected_id}).")
            return False # Indicate failure
    except Exception as e:
        print(f"Error testing microphone {device_id} ({devices[device_id]['name'] if device_id < len(devices) else 'Unknown'}): {e}")
        return False # Indicate failure

# --- Performance Settings ---
FRAME_SKIP_RATE = 8  # Process only every N-th frame (higher value = better performance, less responsive)
# LLM_MIN_TEXT_LENGTH = 15  # Minimum text length to send to LLM (ensures meaningful commands) - Removed per user request

# Add these imports at the top near the other audio processing imports
try:
    import scipy.signal
    from scipy import signal
    SCIPY_AVAILABLE = True
    print("✓ SciPy imported successfully for noise filtering.")
except ImportError:
    print("SciPy not found. Advanced noise filtering will be limited. Run: pip install scipy")
    SCIPY_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    print("✓ NoiseReduce imported successfully for noise filtering.")
except ImportError:
    print("NoiseReduce not found. Advanced noise reduction unavailable. Run: pip install noisereduce")
    NOISEREDUCE_AVAILABLE = False

# Add these new functions after the import section but before the voice_listener_thread_func

def denoise_audio(audio_data, sample_rate=16000):
    """Apply noise reduction to audio data"""
    try:
        # Convert to numpy array
        pcm_data = np.frombuffer(audio_data, dtype=AUDIO_DTYPE)
        
        # Apply noise reduction if available
        if NOISEREDUCE_AVAILABLE:
            # Convert to float for noisereduce
            float_data = pcm_data.astype(np.float32) / 32768.0
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=float_data,
                sr=sample_rate,
                stationary=False,  # Non-stationary for varying environments
                prop_decrease=0.75
            )
            
            # Convert back to int16
            result = (reduced_noise * 32768.0).astype(np.int16)
            print(f"Applied noisereduce: in_energy={np.mean(np.abs(pcm_data)):.1f}, out_energy={np.mean(np.abs(result)):.1f}")
            return result.tobytes()
        else:
            return audio_data
    except Exception as e:
        print(f"Error in noise reduction: {e}")
        return audio_data

def apply_bandpass_filter(audio_data, sample_rate=16000):
    """Apply bandpass filter to focus on human voice frequencies (300-3400 Hz)"""
    try:
        if not SCIPY_AVAILABLE:
            return audio_data
            
        # Convert to numpy array
        pcm_data = np.frombuffer(audio_data, dtype=AUDIO_DTYPE)
        
        # Human voice is typically between 300-3400 Hz
        # Design bandpass filter
        nyquist = 0.5 * sample_rate
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.lfilter(b, a, pcm_data)
        
        print(f"Applied bandpass filter: in_energy={np.mean(np.abs(pcm_data)):.1f}, out_energy={np.mean(np.abs(filtered_data)):.1f}")
        
        # Return as bytes
        return filtered_data.astype(AUDIO_DTYPE).tobytes()
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return audio_data

def calculate_audio_snr(audio_data):
    """Calculate signal-to-noise ratio of audio data"""
    try:
        pcm_data = np.frombuffer(audio_data, dtype=AUDIO_DTYPE)
        
        # Estimate signal and noise
        # This is a simple approach - in a real system you'd use a more sophisticated method
        signal_level = np.max(np.abs(pcm_data))
        noise_level = np.mean(np.abs(pcm_data))
        
        # Calculate SNR, avoiding division by zero
        if noise_level > 0:
            snr = 20 * np.log10(signal_level / noise_level)
        else:
            snr = 0
            
        return snr
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return 0

if __name__ == "__main__":
    print("\n=====================================================")
    print("Hand Gesture and Voice Control System Initializing...")
    print("=====================================================")
    
    # Check voice status
    print("\n🎤 VOICE STATUS:")
    if VOICE_ENABLED:
        print("Voice recognition is ENABLED")
        print(f"Using remote Whisper server: {WHISPER_SERVER_URL}")
        print(f"Default model: {MODEL_NAME}")
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
        
        # List available audio devices
        try:
            print("\nAvailable audio input devices:")
            devices = sd.query_devices()
            input_devices = []
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    input_devices.append((i, dev['name']))
                    print(f"  [{i}] {dev['name']}")
            
            if input_devices:
                default_device = sd.query_devices(kind='input')
                print(f"Default input device: [{default_device['index']}] {default_device['name']}")
            else:
                print("No input devices found!")
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            
        print("Starting voice listener thread...")
        voice_thread = threading.Thread(target=voice_listener_thread_func, daemon=True)
        voice_thread.start()
        print("Voice listener thread started.")
        
        # Start LLM processing thread
        llm_thread = threading.Thread(target=llm_processing_thread_func, daemon=True)
        llm_thread.start()
        print("LLM processing thread started.")
    else:
        print("Voice recognition is DISABLED due to missing libraries or initialization errors")
        print("Please check README.md for setup instructions")

    print("\n🤖 ROBOT CONTROL STATUS:")
    # Load Robot IP before printing its status
    load_robot_ip()
    print(f"Robot IP: {ROBOT_IP}:{ROBOT_PORT}")
    print(f"Initial rotation angle: {current_rotation_angle}°")
    print(f"Initial up-down angle: {current_updown_angle}°")
    print(f"Initial gripper angle: {current_gripper_angle}°")
    
    if recognizer is None:
        print("\n❌ CRITICAL ERROR: Gesture recognizer failed to initialize. Exiting.")
        cv2.destroyAllWindows()
        exit()

    # Simple camera initialization (original approach that was working)
    print("\n🎥 CAMERA STATUS:")
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Simple initialization - no DirectShow, no fancy params
    
    if not cap.isOpened():
        print("❌ ERROR: Failed to open camera. Trying camera index 1...")
        cap = cv2.VideoCapture(1)  # Try index 1 as fallback
        
        if not cap.isOpened():
            print("❌ ERROR: Failed to open any camera. Exiting.")
            cv2.destroyAllWindows()
            exit()
    
    # Read a test frame to confirm camera works
    success, test_frame = cap.read()
    if not success or test_frame is None or test_frame.size == 0:
        print("❌ ERROR: Camera opened but failed to read frames. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
    print("✓ Camera initialized successfully!")
        
    tracking_mode = "GESTURE" 
    current_frame_number = 0
    
    # Initialize gesture variables to avoid undefined variable errors when frames are skipped
    display_gesture = "None"
    stable_gesture = "None"
    custom_gesture = "None"
    recognized_landmarks = None
    gripper_angle = current_gripper_angle
    gripper_status = "GRIPPER_IDLE"
    gripper_cmd_sent = False
    rotation_angle = current_rotation_angle
    rotation_status = "ROT_IDLE"
    rotation_cmd_sent = False
    updown_angle = current_updown_angle
    updown_status = "UPDOWN_IDLE"
    updown_cmd_sent = False
    thumb_px = None
    index_px = None
    dist_px = None
    gesture_debug = None

    # Initialize arm controller if text_to_arm is available
    if TEXT_TO_ARM_AVAILABLE:
        try:
            # Try to load OpenAI API key from environment or .env file
            api_key_available = load_openai_api_key()
            
            if api_key_available:
                print("Initializing arm controller with OpenAI API...")
                arm_controller = LLMController()
                print("✓ Arm controller initialized successfully")
            else:
                print("❌ OpenAI API key not found. Voice commands won't control the arm.")
                print("   See above instructions to set up your OpenAI API key.")
        except Exception as e:
            print(f"❌ Error initializing arm controller: {e}")
            import traceback
            traceback.print_exc()
            TEXT_TO_ARM_AVAILABLE = False
            print("   Check if the OpenAI API key is valid and has sufficient permissions.")

    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Hand Gesture and Voice Control System")
    parser.add_argument(
        "--stt_service", 
        type=str, 
        default="whisper", 
        choices=["whisper", "azure"],
        help="Specify the Speech-to-Text service to use: 'whisper' (default) or 'azure'."
    )
    # Add new arguments for noise thresholds
    parser.add_argument(
        "--noise_threshold",
        type=int,
        default=DEFAULT_NOISE_ENERGY_THRESHOLD,
        help=f"Energy threshold for noise filtering (higher values for noisier environments, default: {DEFAULT_NOISE_ENERGY_THRESHOLD})"
    )
    parser.add_argument(
        "--min_snr",
        type=float,
        default=DEFAULT_MIN_VOICE_SNR,
        help=f"Minimum signal-to-noise ratio to consider as speech (higher values for stricter filtering, default: {DEFAULT_MIN_VOICE_SNR})"
    )
    args = parser.parse_args()
    
    # Apply command-line arguments
    STT_SERVICE = args.stt_service
    NOISE_ENERGY_THRESHOLD = args.noise_threshold
    MIN_VOICE_SNR = args.min_snr

    if STT_SERVICE == "azure" and not AZURE_SDK_AVAILABLE:
        print("❌ Azure STT service selected, but SDK is not available. Please install it.")
        print("Falling back to Whisper STT service.")
        STT_SERVICE = "whisper"
    
    print(f"Selected STT Service: {STT_SERVICE.upper()}")
    print(f"Noise Energy Threshold: {NOISE_ENERGY_THRESHOLD}")
    print(f"Minimum Voice SNR: {MIN_VOICE_SNR}")
    # --- End Command-Line Argument Parsing ---

    while cap.isOpened() and recognizer is not None: 
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera. Check if camera is still connected and not in use by another app.")
            time.sleep(0.5) # Wait a bit before trying again or breaking
            continue # Try to read next frame
        
        current_frame_number += 1
        
        # Frame skipping for performance - only process every Nth frame
        do_process_frame = (current_frame_number % FRAME_SKIP_RATE == 0)
        
        # Always flip frame for display, even if skipping processing
        frame = cv2.flip(frame, 1)
        
        # Only process for gesture recognition if not skipping this frame
        if do_process_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

            (display_gesture, stable_gesture, custom_gesture, recognized_landmarks, 
            gripper_angle, gripper_status, gripper_cmd_sent,
            rotation_angle, rotation_status, rotation_cmd_sent,
            updown_angle, updown_status, updown_cmd_sent,
            thumb_px, index_px, dist_px, gesture_debug) = process_recognized_gestures(frame.shape)
        
            # Only process recognized landmarks if we have them (from this frame or previous)
            if recognized_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else 0.0,
                        presence=lm.presence if hasattr(lm, 'presence') else 0.0
                    ) for lm in recognized_landmarks
                ])
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if DEBUG_GESTURES and gesture_debug and recognized_landmarks:
                    h, w = frame.shape[:2]
                    wrist = recognized_landmarks[mp.solutions.hands.HandLandmark.WRIST]
                    index_mcp = recognized_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
                    index_tip = recognized_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    pinky_mcp = recognized_landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP]
                    
                    # Draw wrist-to-index base line (hand axis)
                    wrist_px = (int(wrist.x * w), int(wrist.y * h))
                    index_mcp_px = (int(index_mcp.x * w), int(index_mcp.y * h))
                    index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
                    pinky_mcp_px = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
                    
                    # Draw hand orientation axis
                    cv2.line(frame, pinky_mcp_px, index_mcp_px, (255, 0, 255), 2)  # Magenta line for hand orientation
                    
                    # Draw pointing direction guide
                    cv2.line(frame, index_mcp_px, index_tip_px, (0, 255, 0), 2)  # Green line for pointing direction
                    
                    # Draw debug values
                    direction_value = gesture_debug.get("pointing_direction_x", 0)
                    hand_width = gesture_debug.get("hand_width", 0)
                    scaled_threshold = gesture_debug.get("scaled_threshold", 0)
                    angle = gesture_debug.get("angle_from_vertical", 0)
                    is_palm = gesture_debug.get("is_palm_like", False)
                    
                    debug_start_y = index_tip_px[1]
                    cv2.putText(frame, f"Direction: {direction_value:.2f}", (10, debug_start_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Hand width: {hand_width:.2f}", (10, debug_start_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Threshold: {scaled_threshold:.2f}", (10, debug_start_y + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Angle: {angle:.1f}°", (10, debug_start_y + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Palm-like: {is_palm}", (10, debug_start_y + 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Highlight threshold area with scaled threshold
                    threshold_right_px = index_mcp_px[0] + int(scaled_threshold * w)
                    threshold_left_px = index_mcp_px[0] - int(scaled_threshold * w)
                    
                    # Draw threshold visualization
                    cv2.line(frame, (threshold_right_px, index_mcp_px[1] - 30), 
                             (threshold_right_px, index_mcp_px[1] + 30), (0, 165, 255), 1)  # Right threshold
                    cv2.line(frame, (threshold_left_px, index_mcp_px[1] - 30), 
                             (threshold_left_px, index_mcp_px[1] + 30), (0, 165, 255), 1)   # Left threshold

                if thumb_px and index_px: 
                    cv2.line(frame, thumb_px, index_px, (255, 0, 0), 2)
                    mid_point = ((thumb_px[0] + index_px[0]) // 2, 
                                 (thumb_px[1] + index_px[1]) // 2)
                    cv2.putText(frame, f"{dist_px:.1f}px", mid_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.putText(frame, f"Detected: {display_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Stable: {stable_gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if stable_gesture != "None" else (0,0,255), 2)
        
        if custom_gesture != "None":
            cv2.putText(frame, f"Custom: {custom_gesture}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Orange color for custom gestures

        # Display all supported gestures and highlight the recognized one
        text_y_offset = frame.shape[0] - (len(SUPPORTED_GESTURES) + len(CUSTOM_GESTURES) + 2) * 20 - 10 
        
        cv2.putText(frame, "Standard Gestures:", (10, text_y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y_offset += 20
        
        for gesture_name_to_display in SUPPORTED_GESTURES:
            color = (0, 255, 0) if gesture_name_to_display == stable_gesture and stable_gesture != "None" else (200, 200, 200)
            if gesture_name_to_display == "None" and stable_gesture == "None": 
                color = (0,165,255) 
            cv2.putText(frame, gesture_name_to_display, (10, text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y_offset += 20
        
        text_y_offset += 10  # Extra spacing
        cv2.putText(frame, "Custom Gestures:", (10, text_y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y_offset += 20
        
        for gesture_name_to_display in CUSTOM_GESTURES:
            color = (255, 165, 0) if gesture_name_to_display == custom_gesture else (200, 200, 200)
            cv2.putText(frame, gesture_name_to_display, (10, text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y_offset += 20

        # Display control information for gripper, rotation, and up-down movement
        display_y = 120
        
        gripper_color = (0, 255, 0) if gripper_cmd_sent else (0, 165, 255)
        cv2.putText(frame, f"Gripper: {int(gripper_angle)}° - {gripper_status}", (10, display_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 1)
        display_y += 30
        
        rotation_color = (0, 255, 0) if rotation_cmd_sent else (0, 165, 255)
        cv2.putText(frame, f"Rotation (M1): {int(rotation_angle)}° - {rotation_status}", (10, display_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rotation_color, 1)
        display_y += 30
        
        updown_color = (0, 255, 0) if updown_cmd_sent else (0, 165, 255)
        cv2.putText(frame, f"Up-Down (M2): {int(updown_angle)}° - {updown_status}", (10, display_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, updown_color, 1)
        
        # Gripper visualization
        frame = draw_gripper_visualization(frame, gripper_angle)
        
        # Display mode and robot info
        cv2.putText(frame, f"Mode: {tracking_mode}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Performance info
        perf_text = f"Frame Skip: {FRAME_SKIP_RATE}x" + (f" [Processing]" if do_process_frame else " [Skipping]")
        cv2.putText(frame, perf_text, (frame.shape[1] - 240, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        cv2.putText(frame, f"Robot: {ROBOT_IP}:{ROBOT_PORT}", (frame.shape[1] - 240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Process any voice commands from the queue
        if VOICE_ENABLED:
            try:
                voice_command = voice_command_queue.get_nowait()
                print(f"Processing voice command from queue: {voice_command}")
                
                # Use text_to_arm functionality if available
                if TEXT_TO_ARM_AVAILABLE and arm_controller:
                    try:
                        print(f"Sending voice command to arm controller: '{voice_command}'")
                        threading.Thread(target=lambda: arm_controller.execute_text_command(voice_command), daemon=True).start()
                        last_transcribed_text = f"Executing: {voice_command[:30]}..." # Show it on screen briefly
                    except Exception as e:
                        print(f"Error processing command with arm controller: {e}")
                        last_transcribed_text = f"Error: {str(e)[:30]}..."
                else:
                    # Original placeholder behavior
                    last_transcribed_text = f"CMD: {voice_command[:30]}... (Arm control not available)"
                    
            except queue.Empty:
                pass # No new voice command

        # Display voice status and last transcribed text
        if VOICE_ENABLED:
            # Get and display active microphone name
            try:
                devices = sd.query_devices()
                default_input = sd.query_devices(kind='input')
                mic_name = default_input['name']
                mic_info = f"Mic: {mic_name[:20]}..." if len(mic_name) > 20 else f"Mic: {mic_name}"
            except Exception as e:
                mic_info = "Mic: Unknown"
            
            # Display voice status with color based on state
            status_color = (255, 255, 0)  # Default yellow
            if "Hearing voice" in voice_status:
                status_color = (0, 255, 0)  # Green when hearing voice
            elif "Processing" in voice_status:
                status_color = (0, 165, 255)  # Orange when processing
            elif "waiting for response" in voice_status:
                status_color = (0, 255, 255)  # Yellow when waiting for LLM
            elif "Error" in voice_status or "disabled" in voice_status.lower():
                status_color = (0, 0, 255)  # Red for errors
            
            cv2.putText(frame, f"Voice: {voice_status}", (10, frame.shape[0] - 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            cv2.putText(frame, mic_info, (10, frame.shape[0] - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            
            # Show LLM status
            if TEXT_TO_ARM_AVAILABLE and arm_controller:
                llm_status = "LLM/Arm: Connected & Ready"
                llm_color = (0, 255, 0)  # Green for ready
            else:
                llm_status = "LLM/Arm: Not available - OpenAI API key needed"
                llm_color = (0, 0, 255)  # Red for not available
            
            cv2.putText(frame, llm_status, (10, frame.shape[0] - 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, llm_color, 1)
            
            # Display accumulated text with a different color
            if accumulated_text:
                # Get remaining time before sending to LLM (only show if not waiting for response)
                if not waiting_for_llm_response:
                    silence_duration_ms = (time.time() - last_speech_end_time) * 1000
                    remaining_ms = max(0, LLM_SILENCE_MS - silence_duration_ms)
                    
                    # Show accumulated text with accumulating indication
                    collection_text = f"Collecting: {accumulated_text}"
                    cv2.putText(frame, collection_text, (10, frame.shape[0] - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 1)  # Green for accumulated text
                    
                    # Draw the silence timer progress bar
                    cv2.rectangle(frame, (10, frame.shape[0] - 45), (210, frame.shape[0] - 35), (100, 100, 100), 1)
                    time_progress = int((remaining_ms / LLM_SILENCE_MS) * 200)
                    cv2.rectangle(frame, (10, frame.shape[0] - 45), (10 + time_progress, frame.shape[0] - 35), (50, 205, 50), -1)
                    
                    cv2.putText(frame, f"Sending in {remaining_ms/1000:.1f}s", (220, frame.shape[0] - 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 205, 50), 1)
            
            # Show waiting for LLM indicator when appropriate
            if waiting_for_llm_response:
                wait_text = "⏳ Waiting for LLM to process command..."
                cv2.putText(frame, wait_text, (10, frame.shape[0] - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)  # Orange for waiting
            
            if last_transcribed_text:
                cv2.putText(frame, f"Heard: {last_transcribed_text}", (10, frame.shape[0] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        # Display error message if there is one and it's within the display duration
        if error_message and (time.time() - error_message_time) < ERROR_DISPLAY_DURATION:
            # Draw a semi-transparent overlay in the center of the screen
            overlay = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
            
            # Add text to the overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_color = (0, 0, 255)  # Red
            line_height = 30
            
            # Split message into multiple lines if needed
            max_width = w//2
            words = error_message.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                text_size = cv2.getTextSize(test_line, font, font_scale, 2)[0]
                
                if text_size[0] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            y_pos = h//3 + 50
            for line in lines:
                cv2.putText(overlay, line, (w//4 + 20, y_pos), font, font_scale, text_color, 2)
                y_pos += line_height
            
            # Add transparency to the overlay
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
        cv2.imshow("Hand Gesture Robot Control", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # Main loop ends here

    if recognizer:
        recognizer.close()
    cap.release()
    cv2.destroyAllWindows()
