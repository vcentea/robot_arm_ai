import os
import time
import sys
import numpy as np
import json
from typing import Optional, List

# FastAPI for server functionality
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Audio processing
from pydantic import BaseModel
import torch
from io import BytesIO

# Silence specific warnings
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Add explicit CUDA check with detailed output
print("Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"✓ CUDA is available! Found {torch.cuda.device_count()} device(s)")
    print(f"✓ Primary GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ CUDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"✓ CUDNN version: {torch.backends.cudnn.version()}")
else:
    print("❌ CUDA not available! Will use CPU but performance will be much slower.")
    print("   Please check NVIDIA drivers and PyTorch CUDA installation.")

# Whisper - use faster-whisper instead of original OpenAI whisper
try:
    print("Attempting to import faster-whisper...")
    from faster_whisper import WhisperModel
    print("✓ faster-whisper imported successfully.")
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("faster-whisper not found. Please install: pip install faster-whisper")
    WHISPER_AVAILABLE = False

# --- Configuration ---
PORT = 5555
HOST = "0.0.0.0"  # Listen on all interfaces
MODEL_NAME = "large-v3"  # Default model, can be changed
AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
USE_GPU = torch.cuda.is_available()
# Force GPU usage if available
if USE_GPU:
    device = "cuda"
    # Optimize for RTX 3090 Ti
    COMPUTE_TYPE = "float16"  # float16 is optimal for modern NVIDIA GPUs
    torch.backends.cudnn.benchmark = True  # Enable benchmark mode for optimized performance
else:
    device = "cpu"
    COMPUTE_TYPE = "int8"

print(f"Using device: {device}, compute type: {COMPUTE_TYPE}")

# Global variables
asr_model = None
models_cache = {}  # Cache loaded models

app = FastAPI(title="Whisper Transcription Server", 
              description="API server for faster-whisper speech-to-text transcription")

# Data models
class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: float
    model_used: str
    segments: Optional[List[dict]] = None
    gpu_used: bool
    compute_type: str

class ModelListResponse(BaseModel):
    available_models: List[str]
    current_model: str
    gpu_available: bool
    compute_type: str
    
def initialize_model(model_name=MODEL_NAME):
    """Initialize faster-whisper model with specified name"""
    global asr_model, models_cache
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper library not available")
    
    if model_name in models_cache:
        print(f"Using cached model: {model_name}")
        asr_model = models_cache[model_name]
        return asr_model
    
    try:
        print(f"Loading faster-whisper model: {model_name}")
        print("(This may take a moment to download if it's the first time...)")
        start_time = time.time()
        
        # Get device - use cuda if available
        device = "cuda" if USE_GPU else "cpu"
        
        # Create model with CTranslate2 backend for faster inference
        asr_model = WhisperModel(
            model_name, 
            device=device, 
            compute_type=COMPUTE_TYPE,
            cpu_threads=min(8, os.cpu_count() or 4),  # Optimize CPU thread count
            download_root=os.path.join(os.getcwd(), "models")  # Save models in a specific directory
        )
        
        elapsed = time.time() - start_time
        print(f"✓ faster-whisper model '{model_name}' loaded successfully in {elapsed:.1f} seconds!")
        print(f"Using device: {device}, compute type: {COMPUTE_TYPE}")
        
        # Print GPU memory usage if available
        if USE_GPU:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"GPU memory allocated: {allocated:.2f} GB (max: {max_allocated:.2f} GB)")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Force GPU memory cleanup to ensure maximum available for inference
            torch.cuda.empty_cache()
            print("GPU cache cleared to optimize memory usage")
        
        # Cache the model
        models_cache[model_name] = asr_model
        return asr_model
    except Exception as e:
        print(f"Error initializing faster-whisper model: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    if WHISPER_AVAILABLE:
        try:
            initialize_model()
            print(f"Server ready to accept requests on port {PORT}")
        except Exception as e:
            print(f"Error during startup: {e}")
    else:
        print("WARNING: Server starting without faster-whisper available!")

@app.get("/models", response_model=ModelListResponse)
async def get_models():
    """Get list of available models and current model"""
    return {
        "available_models": AVAILABLE_MODELS,
        "current_model": MODEL_NAME,
        "gpu_available": USE_GPU,
        "compute_type": COMPUTE_TYPE
    }

@app.post("/change_model")
async def change_model(model_name: str = Form(...)):
    """Change the ASR model"""
    global MODEL_NAME, asr_model
    
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=500, detail="faster-whisper not available")
    
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Available models: {AVAILABLE_MODELS}")
    
    try:
        MODEL_NAME = model_name
        asr_model = initialize_model(model_name)
        return {"status": "success", "message": f"Model changed to {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error changing model: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None)
):
    """Transcribe audio file using faster-whisper"""
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=500, detail="faster-whisper not available")
    
    # Use specified model or fallback to default
    model_to_use = model if model in AVAILABLE_MODELS else MODEL_NAME
    
    if asr_model is None or (model is not None and model != MODEL_NAME):
        try:
            initialize_model(model_to_use)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")
    
    try:
        # Read audio file
        start_time = time.time()
        audio_data = await file.read()
        
        # Convert to numpy array
        # We support multiple formats by letting audio libraries handle them
        audio_bytes = BytesIO(audio_data)
        
        # Load audio using soundfile or other method
        try:
            import soundfile as sf
            audio_bytes.seek(0)
            try:
                # Try soundfile for WAV, FLAC, etc.
                data, samplerate = sf.read(audio_bytes)
                # Resample to 16kHz if needed
                if samplerate != 16000:
                    from scipy import signal
                    audio_np = signal.resample(data, int(len(data) * 16000 / samplerate))
                else:
                    audio_np = data
                
                # Convert to mono if stereo
                if len(audio_np.shape) > 1:
                    audio_np = np.mean(audio_np, axis=1)
                
                # Normalize
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32) / (32768.0 if audio_np.dtype == np.int16 else 1.0)
            except:
                # Fallback to raw PCM 16-bit assumption
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as audio_err:
            print(f"Error processing audio: {audio_err}")
            # Last resort fallback
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Clean up GPU memory before and after transcription for optimal performance
        if USE_GPU:
            print("Clearing GPU cache before transcription...")
            torch.cuda.empty_cache()
        
        # Transcribe with faster-whisper - different API than original whisper
        print(f"Starting transcription using model '{model_to_use}' on {device}...")
        segments, info = asr_model.transcribe(
            audio_np, 
            language="en",             # Specify English to speed up detection
            beam_size=5,               # Balanced beam size for accuracy vs. speed
            vad_filter=True,           # Voice activity detection to skip silence
            vad_parameters=dict(min_silence_duration_ms=500)  # VAD settings
        )
        
        # Process segments
        segment_list = []
        full_text = ""
        
        # Convert segments iterator to list and build full text
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                          for word in (segment.words or [])]
            }
            segment_list.append(segment_dict)
            full_text += segment.text + " "
        
        elapsed = time.time() - start_time
        
        # Format response
        response = {
            "text": full_text.strip(),
            "language": info.language,
            "duration": elapsed,
            "model_used": model_to_use,
            "segments": segment_list,
            "gpu_used": USE_GPU,
            "compute_type": COMPUTE_TYPE
        }
        
        # Print performance metrics in console
        audio_duration = len(audio_np)/16000
        speedup = audio_duration/elapsed if elapsed > 0 else 0
        print(f"Transcribed {audio_duration:.2f}s audio in {elapsed:.2f}s " + 
              f"({speedup:.2f}x real-time)")
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy" if WHISPER_AVAILABLE and asr_model is not None else "unhealthy",
        "model_loaded": asr_model is not None,
        "current_model": MODEL_NAME,
        "gpu_available": USE_GPU,
        "compute_type": COMPUTE_TYPE,
        "backend": "faster-whisper (CTranslate2)",
        "time": time.time()
    }
    
    # Add GPU info if available
    if USE_GPU:
        health["cuda_device"] = torch.cuda.get_device_name(0)
        health["cuda_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        health["cuda_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
    
    return health

if __name__ == "__main__":
    print("=== Faster Whisper Transcription Server ===")
    
    # Handle command line arguments for model selection
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg in AVAILABLE_MODELS:
            MODEL_NAME = model_arg
            print(f"Using specified model: {MODEL_NAME}")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"Operating system: {os.name}")
    print(f"GPU available: {USE_GPU}")
    if USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Compute type: {COMPUTE_TYPE}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Start server
    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT) 