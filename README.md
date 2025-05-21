# Robot Vision and Voice Control System

This project implements a hand gesture and voice control system for a robot arm using computer vision and speech recognition.

## Features

- Hand gesture recognition using MediaPipe
- Custom pointing gesture detection 
- Voice command recognition using OpenAI Whisper
- Integrated control for robot gripper, rotation, and up/down movements
- Support for multiple voice recognition model sizes
- Support for LLM integration with detected speech
- Transcription API server for speech-to-text services

## Requirements

### Main Dependencies
- OpenCV
- MediaPipe
- NumPy
- OpenAI Whisper
- SoundDevice
- WebRTCVAD
- Requests

For the transcription server:
- FastAPI
- Uvicorn
- SoundFile
- Pydantic
- Python-multipart

## Voice Recognition Setup

The system uses OpenAI's Whisper for speech recognition. To use this feature:

1. Install the required libraries:
```bash
pip install openai-whisper sounddevice webrtcvad numpy
```

2. The system will automatically download the required model files on first use.

3. Voice commands are processed through a Voice Activity Detection (VAD) system that:
   - Listens continuously for speech
   - Processes audio only when speech is detected
   - Transcribes the speech to text using Whisper
   - Passes the text to the main system for processing

## Whisper Transcription Server

The system includes a standalone transcription server that provides a REST API for speech-to-text:

### Running the Server

```bash
# Install server dependencies
pip install -r requirements.txt

# Start the server on port 5555
python whisper_server.py medium.en
```

The server will automatically use GPU acceleration if available (recommended for medium and large models).

### API Endpoints

- `POST /transcribe` - Transcribe an audio file
- `GET /models` - List available models
- `POST /change_model` - Change the active model
- `GET /health` - Check server health

### Using the Client

A client script is included for testing and demonstration:

```bash
# Transcribe an audio file
python whisper_client.py --file path/to/audio.wav

# Use a specific model
python whisper_client.py --file path/to/audio.wav --model large-v3

# List available models
python whisper_client.py --list-models

# Check server health
python whisper_client.py --health

# Change server model
python whisper_client.py --change-model medium.en
```

## Whisper Model Sizes

The system supports multiple Whisper model sizes, each with different accuracy and resource requirements:

| Model     | Size    | Memory Required | Speed   | Accuracy | Use Case                |
|-----------|---------|-----------------|---------|----------|-------------------------|
| tiny.en   | ~75MB   | 2GB RAM         | Fastest | Lowest   | Quick testing, mobile   |
| base.en   | ~150MB  | 2GB RAM         | Fast    | Low      | Basic applications      |
| small.en  | ~500MB  | 4GB RAM         | Medium  | Medium   | General use             |
| medium.en | ~1.5GB  | 8GB RAM         | Slow    | High     | Professional/important  |
| large-v3  | ~3GB    | 16GB RAM        | Slowest | Highest  | Critical accuracy needs |

You can select the model size when starting the test script:

```bash
python test_openai_whisper.py medium.en
```

Or switch models during runtime:

```bash
# In the test script
> model medium.en
```

## LLM Integration

The system now includes integration points for sending transcribed speech to a language model (LLM) after detecting a longer pause in speech:

1. Short pauses (default: 600ms) trigger immediate transcription for real-time control
2. Longer pauses (default: 2000ms) signal a complete thought that gets sent to the LLM processor
3. The system accumulates consecutive utterances as part of a single thought until a longer pause is detected

The current implementation includes a placeholder for LLM integration.

## Testing Voice Recognition

The repository includes a standalone test script to verify voice recognition:

```bash
python test_openai_whisper.py
```

This script allows you to:
- Test microphone input
- Verify VAD detection
- Test transcription accuracy
- Try continuous listening mode
- Experiment with different Whisper model sizes

## Troubleshooting

If you encounter voice recognition issues:

1. Make sure your microphone is working and properly configured
2. Check that all required packages are installed
3. Verify that the Whisper model downloaded successfully
4. Try using a smaller model like "tiny.en" or "base.en" if you have performance issues
5. Check the console output for detailed error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

# Robot Control System

This project implements various interfaces for controlling a robot arm using speech, vision, and text commands.

## Components

- **Vision Processing**: Object detection and tracking using computer vision
- **Speech Recognition**: Using fast whisper model for speech-to-text conversion
- **Arm Control**: Direct control of a 6-DOF robotic arm
- **Text-to-Arm**: Convert natural language commands to arm movements using LLM

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key (for text-to-arm functionality):
```bash
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On Linux
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Whisper Speech Recognition Server
```bash
python whisper_server.py
```

### Robot Arm Server
```bash
python server/server_arm.py
```

### Text-to-Arm Control
Convert natural language to arm movements:
```bash
python text_to_arm.py [--host ARM_SERVER_HOST] [--port ARM_SERVER_PORT] [--api-key OPENAI_API_KEY]
```

Example commands:
- "Wave hello with the arm"
- "Pick up an object from the table"
- "Move to home position"
- "Rotate the base to face right"

## License

This software is proprietary and confidential. 