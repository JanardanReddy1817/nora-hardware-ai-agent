# NORA — Hardware Voice AI Agent

NORA is a real-time voice-controlled hardware AI prototype built around an ESP32, Gemini Live, local Vosk wake-word detection, and Wokwi simulation.

## Architecture

```text
🎤 Microphone
      ↓
Local Vosk wake-word detection
      ↓
   "Hey Nora"
      ↓
Gemini Live voice agent
      ↓
Tool / function calling
      ↓
Wi-Fi + HTTP
      ↓
ESP32
 ┌────┼─────┐
 ↓    ↓     ↓
LED Buzzer OLED
```

## Features

- Local "Hey Nora" wake-word detection
- Gemini Live voice interaction
- NORA voice responses
- Natural-language hardware commands
- Light ON / OFF
- Light blinking with duration and speed
- Buzzer with configurable count and interval
- ESP32 status query
- Wokwi-based ESP32 simulation
- Microphone echo protection while NORA speaks
- Automatic Live-session recovery / reconnection
- Session resumption support in the Live client

## Example commands

```text
Hey Nora

Turn the light on.

Turn the light on for five seconds.

Blink the light for ten seconds.

Blink the light quickly for five seconds.

Make the buzzer beep three times.

What is the status?

Goodbye Nora
```

## Repository layout

```text
nora-hardware-ai-agent/
├── README.md
├── nora.py
├── wake_test.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── esp32/
    ├── README.md
    └── src/
        └── main.cpp
```

## Software requirements

- Python
- PlatformIO
- Wokwi
- A microphone and speakers
- Gemini API key
- Vosk English speech model

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set the Gemini key using an environment variable:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Never place the API key in source code or commit it to Git.

## Vosk wake model

Place the Vosk model directory next to `nora.py` using the folder name expected by the configuration in the script. The current prototype uses `lyra-model` as the filesystem folder name even though the assistant is named NORA.

## Running NORA

Start the ESP32/Wokwi simulation first, then run:

```powershell
python nora.py
```

Say:

```text
Hey Nora
```

Then give a natural-language hardware command.

## Security

The Gemini API key must stay outside Git. The repository `.gitignore` excludes common local secret files, virtual environments, and local speech-model directories.

If an API key has ever been pasted into chat, source code, screenshots, or Git history, revoke/rotate it before publishing the repository.

## Project status

Prototype / learning project.

Built with Python, Google Gemini Live, Vosk, PlatformIO, ESP32, and Wokwi.
