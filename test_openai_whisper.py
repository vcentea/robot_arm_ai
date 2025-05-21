import time
import sys
import os
import numpy as np
import threading
import queue

try:
    print("Attempting to import whisper (OpenAI version)...")
    import whisper
    print("✓ whisper imported successfully.")

    print("Attempting to import sounddevice...")
    import sounddevice as sd
    print("✓ sounddevice imported successfully.")

    print("Attempting to import webrtcvad...")
    import webrtcvad
    print("✓ webrtcvad imported successfully.")
    
    VOICE_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("One or more voice recognition libraries not found.")
    print("Please install: pip install openai-whisper sounddevice webrtcvad numpy")
    VOICE_LIBS_AVAILABLE = False
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    VOICE_LIBS_AVAILABLE = False

# --- Audio Configuration ---
MODEL_NAME = "tiny.en"  # Default model, can be changed through command line
AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]

VAD_SENSITIVITY = 3  # 0-3 (strict to lenient), 3 is most sensitive
VAD_FRAME_MS = 30    # VAD works in 10, 20, or 30 ms frames
VAD_SILENCE_MS = 700 # How long a pause ends an utterance
LLM_SILENCE_MS = 2000 # Longer pause before sending to LLM (2 seconds)
AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16" # PCM 16-bit
AUDIO_BLOCKSIZE_FRAMES = int(AUDIO_SAMPLERATE * VAD_FRAME_MS / 1000)

# Global variables for status
asr_model = None
vad_instance = None
voice_status = "Initializing..."
transcription_queue = queue.Queue()
llm_queue = queue.Queue()  # Queue for messages to be sent to LLM
last_transcribed_text = ""
accumulated_text = ""  # To accumulate text before sending to LLM
last_speech_end_time = 0  # Track when the last speech segment ended

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

    # 2. Initialize OpenAI Whisper
    try:
        print(f"Loading Whisper model: {MODEL_NAME}")
        print("(This may take a moment to download if it's the first time...)")
        asr_model = whisper.load_model(MODEL_NAME)
        print(f"✓ Whisper model '{MODEL_NAME}' loaded successfully!")
        voice_status = "ASR Initialized. Ready to listen."
        return True
    except Exception as e:
        voice_status = f"Whisper Initialization Error: {e}"
        print(voice_status)
        import traceback
        traceback.print_exc()
        return False

def record_and_transcribe(duration_seconds=5):
    """Record audio for a fixed duration and transcribe it"""
    global voice_status, last_transcribed_text

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
        
        # Convert int16 to float32 and normalize to [-1.0, 1.0] range
        pcm_data_float32 = pcm_data_int16.astype(np.float32) / 32768.0
        
        # Process audio with Whisper
        print("Transcribing audio...")
        result = asr_model.transcribe(pcm_data_float32)
        text = result["text"].strip()

        if text:
            last_transcribed_text = text
            voice_status = f"Heard: {text}"
            print(f"Transcription Result: '{text}'")
            return text
        else:
            voice_status = "No speech detected or transcribed."
            print("No speech detected or transcribed in the recording.")
            return ""

    except Exception as e:
        voice_status = f"Error during recording/transcription: {e}"
        print(voice_status)
        import traceback
        traceback.print_exc()
        return ""
    finally:
        print("--- Recording test complete ---")

def check_and_send_to_llm():
    """Check if enough time has passed to send accumulated text to LLM"""
    global accumulated_text, last_speech_end_time
    
    current_time = time.monotonic()
    if accumulated_text and (current_time - last_speech_end_time) * 1000 > LLM_SILENCE_MS:
        print(f"\n🤖 Sending to LLM: '{accumulated_text}'")
        llm_queue.put(accumulated_text)
        accumulated_text = ""  # Reset after sending

def continuous_listen_thread_func():
    """Thread function that continuously listens for speech and transcribes it"""
    global voice_status, transcription_queue, accumulated_text, last_speech_end_time
    
    if not VOICE_LIBS_AVAILABLE or not asr_model or not vad_instance:
        print("Voice components not available for continuous listening")
        return
    
    # Buffer to collect audio frames flagged as speech
    speech_frames = bytearray()
    utterance_started_time = None
    audio_buffer = queue.Queue()
    
    def _audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        audio_buffer.put(bytes(indata))
    
    try:
        with sd.RawInputStream(samplerate=AUDIO_SAMPLERATE, 
                              blocksize=AUDIO_BLOCKSIZE_FRAMES,
                              dtype=AUDIO_DTYPE, 
                              channels=AUDIO_CHANNELS, 
                              callback=_audio_callback):
            print("🎙️ Voice listener started. Waiting for voice input...")
            voice_status = "Listening..."
            
            while True:
                # Check if we should send accumulated text to LLM
                check_and_send_to_llm()
                
                frame = audio_buffer.get() # Blocks until audio data is available
                is_speech = vad_instance.is_speech(frame, AUDIO_SAMPLERATE)
                current_time = time.monotonic()

                if is_speech:
                    speech_frames += frame
                    if utterance_started_time is None:
                        utterance_started_time = current_time
                        voice_status = "Hearing voice..."
                        print("Speech detected!")
                elif utterance_started_time and (current_time - utterance_started_time > VAD_SILENCE_MS / 1000.0):
                    voice_status = "Processing speech..."
                    print("Processing detected speech...")
                    last_speech_end_time = current_time  # Update the time when speech ended
                    
                    if len(speech_frames) > 0:
                        pcm_data = np.frombuffer(speech_frames, dtype=np.int16)
                        if len(pcm_data) > AUDIO_SAMPLERATE / 2:  # At least 0.5 second of speech
                            try:
                                # Process in separate thread to not block the audio thread
                                def process_audio():
                                    global accumulated_text
                                    try:
                                        # Convert int16 to float32 and normalize to [-1.0, 1.0] range
                                        pcm_data_float32 = pcm_data.astype(np.float32) / 32768.0
                                        
                                        result = asr_model.transcribe(pcm_data_float32)
                                        text = result["text"].strip()
                                        if text:
                                            print(f"👤 Transcribed: {text}")
                                            transcription_queue.put(text)
                                            
                                            # Add to accumulated text for LLM
                                            if accumulated_text:
                                                accumulated_text += " " + text
                                            else:
                                                accumulated_text = text
                                    except Exception as e:
                                        print(f"Error in transcription: {e}")
                                
                                # Start transcription in background
                                threading.Thread(target=process_audio, daemon=True).start()
                            except Exception as e:
                                print(f"Error starting transcription thread: {e}")
                        else:
                            print("Speech too short, ignoring")
                    
                    speech_frames = bytearray()
                    utterance_started_time = None
                    voice_status = "Listening..."
                elif not is_speech and utterance_started_time is None:
                    voice_status = "Listening..."
                    
    except Exception as e:
        print(f"Exception in continuous listener thread: {e}")
        voice_status = f"Thread Error: {e}"

def llm_processing_thread_func():
    """Thread function to process messages sent to the LLM"""
    while True:
        try:
            message = llm_queue.get()
            if message:
                print(f"\n📝 LLM Processing: '{message}'")
                # This is where you would send the message to an actual LLM
                # For now, just simulating a response
                print("🤖 LLM would process this message here")
                time.sleep(1)  # Simulate processing time
                print("✅ LLM processing complete")
            llm_queue.task_done()
        except Exception as e:
            print(f"Error in LLM processing thread: {e}")

def change_model(new_model_name):
    """Change the ASR model to a different size"""
    global MODEL_NAME, asr_model
    
    if new_model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{new_model_name}' not in available models: {AVAILABLE_MODELS}")
        return False
    
    if new_model_name == MODEL_NAME:
        print(f"Already using model '{MODEL_NAME}'")
        return True
    
    print(f"Changing model from '{MODEL_NAME}' to '{new_model_name}'...")
    MODEL_NAME = new_model_name
    
    try:
        print(f"Loading Whisper model: {MODEL_NAME}")
        print("(This may take a moment to download if it's the first time...)")
        start_time = time.time()
        asr_model = whisper.load_model(MODEL_NAME)
        elapsed = time.time() - start_time
        print(f"✓ Whisper model '{MODEL_NAME}' loaded successfully in {elapsed:.1f} seconds!")
        return True
    except Exception as e:
        print(f"Error loading model '{MODEL_NAME}': {e}")
        import traceback
        traceback.print_exc()
        return False

def show_help():
    print("\n--- Available commands ---")
    print("1. 'record' - Record for 5 seconds and transcribe")
    print("2. 'start' - Start continuous listening")
    print("3. 'model [name]' - Change ASR model, e.g., 'model base.en'")
    print(f"   Available models: {', '.join(AVAILABLE_MODELS)}")
    print("4. 'help' - Show this help message")
    print("5. 'exit' - Exit the program")
    print("---------------------------\n")

if __name__ == "__main__":
    print("=== OpenAI Whisper Voice Recognition Test ===")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg in AVAILABLE_MODELS:
            MODEL_NAME = model_arg
            print(f"Using specified model: {MODEL_NAME}")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"Operating system: {os.name}")
    print(f"Current working directory: {os.getcwd()}")
    
    if initialize_voice_components():
        print("\n✓ Voice components initialized successfully!")
        show_help()
        
        continuous_mode = False
        listen_thread = None
        llm_thread = None
        
        # Start LLM processing thread
        llm_thread = threading.Thread(target=llm_processing_thread_func, daemon=True)
        llm_thread.start()
        
        while True:
            try:
                if continuous_mode:
                    # Check for transcribed text from the continuous listener
                    try:
                        text = transcription_queue.get_nowait()
                        print(f"Command received: {text}")
                        # Process the transcribed text as a command if needed
                    except queue.Empty:
                        # No new transcription
                        pass
                
                command = input("\nEnter command (or 'help'): ").strip().lower()
                
                if command == 'exit':
                    print("Exiting...")
                    break
                elif command == 'help':
                    show_help()
                elif command == 'record':
                    record_and_transcribe(duration_seconds=5)
                elif command == 'start':
                    if not continuous_mode:
                        print("Starting continuous listening mode...")
                        continuous_mode = True
                        listen_thread = threading.Thread(target=continuous_listen_thread_func, daemon=True)
                        listen_thread.start()
                    else:
                        print("Continuous listening is already active!")
                elif command.startswith('model '):
                    model_parts = command.split()
                    if len(model_parts) == 2:
                        new_model = model_parts[1]
                        change_model(new_model)
                    else:
                        print(f"Current model: {MODEL_NAME}")
                        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Exiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Test script finished.")
    else:
        print("Could not initialize voice components. Exiting.") 