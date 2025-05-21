import time
import sys
import os
import numpy as np

# Set this to True to print detailed debug information
DEBUG = True

# First try to import libraries with detailed error handling
try:
    print("Attempting to import whispercpp...")
    import whispercpp
    print("whispercpp imported successfully.")

    print("Attempting to import sounddevice...")
    import sounddevice as sd
    print("sounddevice imported successfully.")

    print("Attempting to import webrtcvad...")
    import webrtcvad
    print("webrtcvad imported successfully.")
    
    VOICE_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("One or more voice recognition libraries not found.")
    print("Please ensure you have installed: pywhispercpp sounddevice webrtcvad numpy")
    VOICE_LIBS_AVAILABLE = False
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    VOICE_LIBS_AVAILABLE = False

# --- Audio Configuration ---
MODEL_ID = "tiny.en"  # Using a smaller model which is more likely to work
# MODEL_PATH = "base.en-q5_1.bin"  # Local model file as fallback
VAD_SENSITIVITY = 3  # 0-3 (strict to lenient), 3 is most sensitive
VAD_FRAME_MS = 30    # VAD works in 10, 20, or 30 ms frames
VAD_SILENCE_MS = 700 # How long a pause ends an utterance
AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16" # PCM 16-bit
AUDIO_BLOCKSIZE_FRAMES = int(AUDIO_SAMPLERATE * VAD_FRAME_MS / 1000)

# Global variables for status
asr_model = None
vad_instance = None
voice_status = "Initializing..."

def initialize_voice_components():
    global asr_model, vad_instance, voice_status

    if not VOICE_LIBS_AVAILABLE:
        voice_status = "Voice libraries missing."
        print(voice_status)
        return False

    # 1. Initialize VAD
    try:
        print("Initializing VAD...")
        vad_instance = webrtcvad.Vad(VAD_SENSITIVITY)
        print("VAD initialized successfully.")
    except Exception as e:
        voice_status = f"VAD Initialization Error: {e}"
        print(voice_status)
        return False

    # 2. Try multiple approaches to initialize Whisper
    print("\n--- Trying multiple approaches to initialize Whisper ---")
    
    # Approach 1: Use from_pretrained (most common)
    try:
        print(f"Approach 1: Attempting to load Whisper model using from_pretrained with identifier: '{MODEL_ID}'")
        asr_model = whispercpp.Whisper.from_pretrained(MODEL_ID)
        print("✓ Whisper ASR initialized successfully using from_pretrained!")
        voice_status = "ASR Initialized. Ready to listen."
        return True
    except Exception as e:
        print(f"✗ Approach 1 failed: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
    
    # Approach 2: Try direct constructor (some versions)
    try:
        print(f"\nApproach 2: Attempting direct constructor with model_name parameter")
        asr_model = whispercpp.Whisper(model_name=MODEL_ID)
        print("✓ Whisper ASR initialized successfully using direct constructor!")
        voice_status = "ASR Initialized. Ready to listen."
        return True
    except Exception as e:
        print(f"✗ Approach 2 failed: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
    
    # Approach 3: Try local model file if exists
    local_model = "base.en-q5_1.bin"
    if os.path.exists(local_model):
        try:
            print(f"\nApproach 3: Attempting to load local model file: {local_model}")
            asr_model = whispercpp.Whisper(model_path=local_model)
            print("✓ Whisper ASR initialized successfully using local model file!")
            voice_status = "ASR Initialized. Ready to listen."
            return True
        except Exception as e:
            print(f"✗ Approach 3 failed: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

    # If we get here, all approaches failed
    voice_status = "Failed to initialize Whisper with all approaches"
    print("\n❌ All initialization approaches failed")
    print("This suggests an incompatibility with pywhispercpp on your system.")
    print("Consider using one of these alternatives:")
    print("1. whisper-cpp-python (pip install whisper-cpp-python)")
    print("2. OpenAI's whisper (pip install -U openai-whisper)")
    return False

def record_and_transcribe(duration_seconds=5):
    global voice_status

    if not asr_model or not vad_instance:
        voice_status = "ASR or VAD not initialized."
        print(voice_status)
        return

    print(f"\n--- Starting {duration_seconds}-second recording ---")
    voice_status = f"Listening for {duration_seconds}s..."
    print(voice_status)

    try:
        # Record audio
        audio_frames = sd.rec(int(duration_seconds * AUDIO_SAMPLERATE), 
                              samplerate=AUDIO_SAMPLERATE, 
                              channels=AUDIO_CHANNELS, 
                              dtype=AUDIO_DTYPE)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        voice_status = "Processing audio..."
        print(voice_status)

        pcm_data_int16 = np.array(audio_frames).flatten()
        pcm_data_float32 = pcm_data_int16.astype(np.float32) / 32768.0
        
        print(f"Transcribing {len(pcm_data_float32)/AUDIO_SAMPLERATE:.2f}s of audio...")
        result = asr_model.transcribe(pcm_data_float32) 
        text = result.get("text", "").strip() # Safely get text

        if text:
            voice_status = f"Heard: {text}"
            print(f"Transcription Result: '{text}'")
        else:
            voice_status = "No speech detected or transcribed."
            print("No speech detected or transcribed in the recording.")

    except Exception as e:
        voice_status = f"Error during recording/transcription: {e}"
        print(voice_status)
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        print("--- Recording test complete ---")

if __name__ == "__main__":
    print("--- Voice-to-Text Test Script (with multiple initialization methods) ---")
    
    # Print system information for debugging
    print(f"Python version: {sys.version}")
    print(f"Operating system: {os.name}")
    print(f"Current working directory: {os.getcwd()}")
    
    # List .bin files in current directory
    print("\nChecking for model files in current directory:")
    model_files = [f for f in os.listdir('.') if f.endswith('.bin') or f.endswith('.gguf')]
    if model_files:
        print(f"Found model files: {', '.join(model_files)}")
    else:
        print("No .bin or .gguf model files found in current directory")
    
    if initialize_voice_components():
        try:
            input("Press Enter to start a 5-second test recording...")
            record_and_transcribe(duration_seconds=5)
            
        except KeyboardInterrupt:
            print("\nExiting test script.")
        finally:
            print("Test script finished.")
    else:
        print("Could not initialize voice components. Exiting.") 