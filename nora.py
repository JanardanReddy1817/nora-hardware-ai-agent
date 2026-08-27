import os
import sys
import json
import queue
import asyncio
import threading
import difflib
import requests

import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types


# ============================================================
# NORA CONFIGURATION
# ============================================================

ASSISTANT_NAME = "NORA"

VOSK_MODEL_PATH = "lyra-model"

GEMINI_MODEL = "gemini-3.1-flash-live-preview"

ESP32_URL = "http://localhost:8180/command"

# Gemini Live input/output formats
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

CHANNELS = 1

# Audio block sizes
WAKE_BLOCK_SIZE = 4000
LIVE_BLOCK_SIZE = 1024

# Reconnect settings
MAX_RECONNECTS = 5

# How long to wait between reconnect attempts
MAX_RECONNECT_DELAY = 8


# ============================================================
# PROGRAM STATE
# ============================================================

stop_program = False

# Set while NORA is talking or speaker audio is still draining.
# Microphone input is blocked during this time.
nora_is_speaking = threading.Event()

# Set when the user asks NORA to stop the active session.
exit_active_session = threading.Event()

# Latest resumable Gemini Live session handle.
session_resume_handle = None


# ============================================================
# AUDIO QUEUES
# ============================================================

input_audio_queue = queue.Queue(
    maxsize=40
)

output_audio_queue = queue.Queue(
    maxsize=150
)


# ============================================================
# SPEAKER BUFFER
# ============================================================

speaker_samples = np.empty(
    0,
    dtype=np.int16
)

speaker_buffer_lock = threading.Lock()


# ============================================================
# API KEY
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    print(
        "ERROR: GEMINI_API_KEY is not set."
    )

    print()

    print(
        '$env:GEMINI_API_KEY="YOUR_NEW_GEMINI_API_KEY"'
    )

    sys.exit(1)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# NORA WAKE PHRASES
# ============================================================

WAKE_PHRASES = [
    "nora",
    "norah",
    "nora ah",
    "no ra",

    "hey nora",
    "hey norah",
    "hey nora ah",
    "hey no ra",

    # Common Vosk variations
    "hey flora",
    "he nora",
    "he norah",
    "hey aura"
]


# ============================================================
# LOCAL ACTIVE-SESSION EXIT PHRASES
# ============================================================

EXIT_PHRASES = [
    "exit",
    "quit",
    "stop listening",
    "go to sleep",
    "sleep now",
    "goodbye",
    "goodbye nora",
    "bye nora",
    "nora goodbye",
    "stop listening nora",
    "go to sleep nora"
]


# ============================================================
# STOP AUDIO DEVICES
# ============================================================

def stop_audio_devices():

    try:
        sd.stop()
    except Exception:
        pass


# ============================================================
# CLEAR INPUT QUEUE
# ============================================================

def clear_input_audio():

    while True:

        try:
            input_audio_queue.get_nowait()

        except queue.Empty:
            break


# ============================================================
# CLEAR OUTPUT QUEUE + SPEAKER BUFFER
# ============================================================

def clear_output_audio():

    global speaker_samples

    while True:

        try:
            output_audio_queue.get_nowait()

        except queue.Empty:
            break

    with speaker_buffer_lock:

        speaker_samples = np.empty(
            0,
            dtype=np.int16
        )


# ============================================================
# CHECK SPEAKER BUFFER
# ============================================================

def speaker_audio_pending():

    with speaker_buffer_lock:

        buffer_has_audio = (
            len(speaker_samples) > 0
        )

    return (
        buffer_has_audio
        or not output_audio_queue.empty()
    )


# ============================================================
# WAIT FOR NORA SPEECH TO FINISH
# ============================================================

async def wait_for_speaker_to_finish():

    # Keep microphone disabled.
    nora_is_speaking.set()

    empty_time = 0.0

    while not stop_program:

        if speaker_audio_pending():

            empty_time = 0.0

        else:

            empty_time += 0.03

            if empty_time >= 0.25:

                break

        await asyncio.sleep(
            0.03
        )

    # Tiny acoustic settling time.
    await asyncio.sleep(
        0.10
    )

    # Remove microphone audio that accumulated
    # before/while NORA was speaking.
    clear_input_audio()

    nora_is_speaking.clear()


# ============================================================
# LOAD VOSK
# ============================================================

def load_vosk():

    print(
        "Loading local NORA wake model..."
    )

    try:

        model = Model(
            VOSK_MODEL_PATH
        )

    except Exception as error:

        print()
        print(
            "ERROR: Could not load Vosk model."
        )
        print(error)
        print()

        sys.exit(1)

    grammar = json.dumps(
        WAKE_PHRASES + ["[unk]"]
    )

    recognizer = KaldiRecognizer(
        model,
        INPUT_SAMPLE_RATE,
        grammar
    )

    recognizer.SetWords(
        False
    )

    print(
        "NORA wake model loaded."
    )

    return recognizer


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

def microphone_callback(
    indata,
    frames,
    time_info,
    status
):
    """
    Microphone callback.

    NORA must not hear her own speaker output.
    """

    # --------------------------------------------------------
    # Echo protection
    # --------------------------------------------------------

    if nora_is_speaking.is_set():

        return

    # --------------------------------------------------------
    # Avoid terminal spam from input overflow
    # --------------------------------------------------------

    if status:

        if not status.input_overflow:

            print(
                f"\nMicrophone: {status}",
                flush=True
            )

    # --------------------------------------------------------
    # Queue microphone audio
    # --------------------------------------------------------

    try:

        if input_audio_queue.full():

            try:
                input_audio_queue.get_nowait()
            except queue.Empty:
                pass

        input_audio_queue.put_nowait(
            bytes(indata)
        )

    except (
        queue.Full,
        queue.Empty
    ):

        pass


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(
    text
):

    return (
        text
        .lower()
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace("?", "")
        .replace("!", "")
    )


# ============================================================
# DETECT NORA WAKE WORD
# ============================================================

def detect_nora(
    text
):

    text = normalize_text(
        text
    )

    if not text:

        return False

    exact_matches = {

        "nora",
        "norah",
        "nora ah",
        "no ra",

        "hey nora",
        "hey norah",
        "hey nora ah",
        "hey no ra",

        "hey flora",
        "he nora",
        "he norah",
        "hey aura"
    }

    if text in exact_matches:

        return True

    words = text.split()

    if len(words) >= 2:

        first_word = words[0]

        remaining = " ".join(
            words[1:]
        )

        if first_word in {
            "hey",
            "he",
            "ay",
            "a"
        }:

            candidates = [
                "nora",
                "norah",
                "nora ah",
                "flora",
                "aura"
            ]

            for candidate in candidates:

                score = (
                    difflib.SequenceMatcher(
                        None,
                        remaining,
                        candidate
                    ).ratio()
                )

                if score >= 0.65:

                    return True

    return False


# ============================================================
# DETECT LOCAL EXIT COMMAND
# ============================================================

def detect_exit_command(
    text
):

    text = normalize_text(
        text
    )

    if not text:

        return False

    if text in EXIT_PHRASES:

        return True

    for phrase in EXIT_PHRASES:

        if phrase in text:

            return True

    return False


# ============================================================
# WAIT FOR "HEY NORA"
# ============================================================

def wait_for_nora(
    recognizer
):

    print()
    print(
        "=" * 50
    )
    print(
        "              NORA SLEEP MODE"
    )
    print(
        "=" * 50
    )
    print()
    print(
        'Say "Hey Nora" to activate.'
    )
    print()

    clear_input_audio()
    clear_output_audio()

    nora_is_speaking.clear()

    with sd.RawInputStream(

        samplerate=
            INPUT_SAMPLE_RATE,

        blocksize=
            WAKE_BLOCK_SIZE,

        channels=
            CHANNELS,

        dtype=
            "int16",

        callback=
            microphone_callback

    ):

        while not stop_program:

            audio_data = (
                input_audio_queue.get()
            )

            if recognizer.AcceptWaveform(
                audio_data
            ):

                result = json.loads(
                    recognizer.Result()
                )

                text = result.get(
                    "text",
                    ""
                ).strip()

                if not text:

                    continue

                print(
                    f"Heard: {text}",
                    flush=True
                )

                if detect_nora(
                    text
                ):

                    print()
                    print(
                        "******************************"
                    )
                    print(
                        "       NORA ACTIVATED"
                    )
                    print(
                        "******************************"
                    )
                    print()

                    clear_input_audio()

                    return True

    return False


# ============================================================
# CHECK ESP32
# ============================================================

def check_esp32():

    try:

        response = requests.get(

            ESP32_URL.replace(
                "/command",
                "/"
            ),

            timeout=2
        )

        return (
            response.status_code == 200
        )

    except requests.RequestException:

        return False


# ============================================================
# SEND COMMAND TO ESP32
# ============================================================

def send_esp32_command(
    command,
    params=None
):

    request_params = {
        "cmd": command
    }

    if params:

        request_params.update(
            params
        )

    try:

        response = requests.get(

            ESP32_URL,

            params=
                request_params,

            timeout=5
        )

        if response.status_code == 200:

            return {

                "success":
                    True,

                "message":
                    response.text
            }

        return {

            "success":
                False,

            "message":
                (
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )
        }

    except requests.RequestException as error:

        return {

            "success":
                False,

            "message":
                str(error)
        }


# ============================================================
# NORA SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are NORA, a real-time hardware AI agent controlling
an ESP32.

Your name is NORA.

VOICE BEHAVIOR
--------------
Speak clearly and calmly.

Use short sentences.

Pause naturally between sentences.

Do not speak rapidly.

Use simple words.

Keep hardware confirmations to one short sentence.

Examples:

"Okay. The light is on."

"Okay. Five beeps."

"Blinking for ten seconds."

"The light is off."


AVAILABLE HARDWARE TOOLS
------------------------

1. control_light

Parameters:
- state: "on" or "off"
- duration: seconds


2. blink_light

Parameters:
- duration: total seconds
- interval: milliseconds between ON and OFF


3. make_beep_sound

Parameters:
- count: number of beeps
- interval: milliseconds between beeps


4. get_status

Returns the current ESP32 status.


GENERAL RULES
-------------

- Understand natural language.
- Ignore polite expressions.
- Extract numbers and durations.
- Never invent hardware.
- Use tools for hardware actions.
- Do not merely describe actions.
- Keep responses short.
- Speak slowly and clearly.
- Remain ready for the next command.

If the user wants to stop the active conversation,
sleep, exit, quit, or says goodbye, acknowledge briefly
and stop the active NORA session.
"""


# ============================================================
# GEMINI TOOLS
# ============================================================

TOOLS = [
    {
        "function_declarations": [

            {
                "name":
                    "control_light",

                "description":
                    "Turn the ESP32 light on or off, "
                    "optionally for a specified duration.",

                "parameters": {

                    "type":
                        "object",

                    "properties": {

                        "state": {

                            "type":
                                "string",

                            "enum": [
                                "on",
                                "off"
                            ]
                        },

                        "duration": {

                            "type":
                                "number",

                            "description":
                                "Duration in seconds. "
                                "Use 0 for no timer."
                        }
                    },

                    "required": [
                        "state",
                        "duration"
                    ]
                }
            },

            {
                "name":
                    "blink_light",

                "description":
                    "Blink the ESP32 light repeatedly "
                    "for a specified duration.",

                "parameters": {

                    "type":
                        "object",

                    "properties": {

                        "duration": {

                            "type":
                                "number",

                            "description":
                                "Total blink duration "
                                "in seconds."
                        },

                        "interval": {

                            "type":
                                "integer",

                            "description":
                                "Milliseconds between "
                                "ON and OFF."
                        }
                    },

                    "required": [
                        "duration",
                        "interval"
                    ]
                }
            },

            {
                "name":
                    "make_beep_sound",

                "description":
                    "Make the ESP32 buzzer beep "
                    "one or more times.",

                "parameters": {

                    "type":
                        "object",

                    "properties": {

                        "count": {

                            "type":
                                "integer",

                            "description":
                                "Number of beeps."
                        },

                        "interval": {

                            "type":
                                "integer",

                            "description":
                                "Milliseconds between "
                                "beeps."
                        }
                    },

                    "required": [
                        "count",
                        "interval"
                    ]
                }
            },

            {
                "name":
                    "get_status",

                "description":
                    "Get the current ESP32 status."
            }
        ]
    }
]


# ============================================================
# EXECUTE HARDWARE TOOL
# ============================================================

def execute_tool(
    function_call
):

    name = function_call.name

    args = (
        function_call.args
        or {}
    )

    print()
    print(
        f"NORA TOOL -> {name}"
    )
    print(
        f"Arguments -> {args}"
    )

    # --------------------------------------------------------
    # LIGHT
    # --------------------------------------------------------

    if name == "control_light":

        state = str(
            args.get(
                "state",
                "off"
            )
        ).lower()

        duration = float(
            args.get(
                "duration",
                0
            )
        )

        duration = max(
            0,
            min(
                duration,
                3600
            )
        )

        result = send_esp32_command(

            "control light",

            {
                "state":
                    state,

                "duration":
                    str(duration)
            }
        )

    # --------------------------------------------------------
    # BLINK
    # --------------------------------------------------------

    elif name == "blink_light":

        duration = float(
            args.get(
                "duration",
                5
            )
        )

        interval = int(
            args.get(
                "interval",
                500
            )
        )

        duration = max(
            0.1,
            min(
                duration,
                3600
            )
        )

        interval = max(
            100,
            min(
                interval,
                2000
            )
        )

        result = send_esp32_command(

            "blink light",

            {
                "duration":
                    str(duration),

                "interval":
                    str(interval)
            }
        )

    # --------------------------------------------------------
    # BEEP
    # --------------------------------------------------------

    elif name == "make_beep_sound":

        count = int(
            args.get(
                "count",
                1
            )
        )

        interval = int(
            args.get(
                "interval",
                300
            )
        )

        count = max(
            1,
            min(
                count,
                20
            )
        )

        interval = max(
            100,
            min(
                interval,
                2000
            )
        )

        result = send_esp32_command(

            "make beep sound",

            {
                "count":
                    str(count),

                "interval":
                    str(interval)
            }
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    elif name == "get_status":

        result = send_esp32_command(
            "status"
        )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        result = {

            "success":
                False,

            "message":
                f"Unknown tool: {name}"
        }

    if result["success"]:

        print(
            "ESP32 -> SUCCESS"
        )

        print(
            f"ESP32 -> {result['message']}"
        )

    else:

        print(
            f"ESP32 -> FAILED: "
            f"{result['message']}"
        )

    return result


# ============================================================
# SEND MICROPHONE AUDIO
# ============================================================

async def audio_sender(
    session
):

    try:

        while (
            not stop_program
            and not exit_active_session.is_set()
        ):

            # NORA is speaking: don't send microphone audio.
            if nora_is_speaking.is_set():

                await asyncio.sleep(
                    0.02
                )

                continue

            audio_data = await asyncio.to_thread(
                input_audio_queue.get
            )

            if (
                stop_program
                or exit_active_session.is_set()
            ):

                break

            # Echo protection re-check.
            if nora_is_speaking.is_set():

                continue

            await session.send_realtime_input(

                audio=types.Blob(

                    data=
                        audio_data,

                    mime_type=
                        "audio/pcm;rate=16000"
                )
            )

    except asyncio.CancelledError:

        pass

    except Exception as error:

        if not stop_program:

            print(
                f"\nNORA audio error: "
                f"{error}",
                flush=True
            )

            raise


# ============================================================
# SPEAKER OUTPUT CALLBACK
# ============================================================

def output_callback(
    outdata,
    frames,
    time_info,
    status
):

    global speaker_samples

    if status:

        # Ignore normal underflow/overflow messages.
        if not (
            status.output_underflow
            or status.output_overflow
        ):

            print(
                f"\nSpeaker: {status}",
                flush=True
            )

    required_samples = (
        frames
        * CHANNELS
    )

    # --------------------------------------------------------
    # Get new Gemini audio chunks.
    # --------------------------------------------------------

    new_chunks = []

    while True:

        try:

            chunk = (
                output_audio_queue
                .get_nowait()
            )

        except queue.Empty:

            break

        if chunk:

            new_chunks.append(
                chunk
            )

    # --------------------------------------------------------
    # Convert raw PCM bytes to int16 samples.
    # --------------------------------------------------------

    if new_chunks:

        try:

            new_audio = np.frombuffer(

                b"".join(
                    new_chunks
                ),

                dtype=np.int16
            )

            if new_audio.size > 0:

                with speaker_buffer_lock:

                    speaker_samples = (
                        np.concatenate(
                            (
                                speaker_samples,
                                new_audio
                            )
                        )
                    )

        except Exception as error:

            print(
                f"\nSpeaker conversion error: "
                f"{error}",
                flush=True
            )

    # --------------------------------------------------------
    # Prepare output block.
    # --------------------------------------------------------

    with speaker_buffer_lock:

        available_samples = len(
            speaker_samples
        )

        samples_to_copy = min(
            available_samples,
            required_samples
        )

        output = np.zeros(
            required_samples,
            dtype=np.int16
        )

        if samples_to_copy > 0:

            output[
                :samples_to_copy
            ] = (
                speaker_samples[
                    :samples_to_copy
                ]
            )

            speaker_samples = (
                speaker_samples[
                    samples_to_copy:
                ]
            )

    # --------------------------------------------------------
    # Write output.
    # --------------------------------------------------------

    if CHANNELS == 1:

        outdata[:, 0] = output

    else:

        outdata[:] = output.reshape(
            frames,
            CHANNELS
        )


# ============================================================
# RECEIVE GEMINI RESPONSES
# ============================================================

async def response_receiver(
    session
):

    global session_resume_handle

    try:

        while (
            not stop_program
            and not exit_active_session.is_set()
        ):

            # ------------------------------------------------
            # Receive one complete model turn.
            #
            # The Python SDK documents receive() as yielding
            # a complete model turn. We then return to receive()
            # again on the SAME Live session.
            # ------------------------------------------------

            async for response in (
                session.receive()
            ):

                if (
                    stop_program
                    or exit_active_session.is_set()
                ):

                    break

                # --------------------------------------------
                # SESSION RESUMPTION UPDATE
                # --------------------------------------------

                if response.session_resumption_update:

                    update = (
                        response
                        .session_resumption_update
                    )

                    if (
                        update.resumable
                        and update.new_handle
                    ):

                        session_resume_handle = (
                            update.new_handle
                        )

                # --------------------------------------------
                # TOOL CALL
                # --------------------------------------------

                if response.tool_call:

                    function_responses = []

                    for function_call in (
                        response
                        .tool_call
                        .function_calls
                    ):

                        result = await asyncio.to_thread(
                            execute_tool,
                            function_call
                        )

                        function_responses.append(

                            types.FunctionResponse(

                                id=
                                    function_call.id,

                                name=
                                    function_call.name,

                                response={

                                    "success":
                                        result[
                                            "success"
                                        ],

                                    "message":
                                        result[
                                            "message"
                                        ]
                                }
                            )
                        )

                    await session.send_tool_response(
                        function_responses=
                            function_responses
                    )

                # --------------------------------------------
                # SERVER CONTENT
                # --------------------------------------------

                if not response.server_content:

                    continue

                content = (
                    response.server_content
                )

                # --------------------------------------------
                # USER TRANSCRIPTION
                # --------------------------------------------

                if content.input_transcription:

                    text = (
                        content
                        .input_transcription
                        .text
                    )

                    if text:

                        print(
                            f"\nYou: {text}",
                            flush=True
                        )

                        # -------------------------------
                        # Local exit command
                        # -------------------------------

                        if detect_exit_command(
                            text
                        ):

                            exit_active_session.set()

                            nora_is_speaking.clear()

                            clear_input_audio()

                            print(
                                "\nNORA going to sleep...",
                                flush=True
                            )

                            break

                # --------------------------------------------
                # MODEL AUDIO
                # --------------------------------------------

                if content.model_turn:

                    # Mute microphone BEFORE buffering
                    # the response audio.
                    nora_is_speaking.set()

                    for part in (
                        content
                        .model_turn
                        .parts
                    ):

                        if part.inline_data:

                            audio_data = (
                                part.inline_data.data
                            )

                            if audio_data:

                                try:

                                    output_audio_queue.put_nowait(
                                        audio_data
                                    )

                                except queue.Full:

                                    try:
                                        output_audio_queue.get_nowait()
                                    except queue.Empty:
                                        pass

                                    try:

                                        output_audio_queue.put_nowait(
                                            audio_data
                                        )

                                    except queue.Full:

                                        pass

                # --------------------------------------------
                # NORA TEXT TRANSCRIPTION
                # --------------------------------------------

                if content.output_transcription:

                    text = (
                        content
                        .output_transcription
                        .text
                    )

                    if text:

                        print(
                            f"Nora: {text}",
                            flush=True
                        )

                # --------------------------------------------
                # TURN COMPLETE
                # --------------------------------------------

                if content.turn_complete:

                    # Wait until actual speaker playback
                    # has drained before reopening microphone.
                    await wait_for_speaker_to_finish()

                    if not exit_active_session.is_set():

                        print(
                            "[Nora ready]",
                            flush=True
                        )

                    break

            # ------------------------------------------------
            # Return to session.receive() for the next turn.
            # DO NOT reconnect here.
            # ------------------------------------------------

            if (
                not stop_program
                and not exit_active_session.is_set()
            ):

                await asyncio.sleep(
                    0.01
                )


    except asyncio.CancelledError:

        pass

    except Exception:

        # Let active_nora() handle the real connection error.
        raise


# ============================================================
# CREATE / RESUME GEMINI LIVE SESSION
# ============================================================

async def run_nora_live(
    resume_handle=None
):

    global session_resume_handle

    config = types.LiveConnectConfig(

        response_modalities=[
            "AUDIO"
        ],

        system_instruction=
            types.Content(

                parts=[
                    types.Part(
                        text=
                            SYSTEM_INSTRUCTIONS
                    )
                ]
            ),

        tools=
            TOOLS,

        realtime_input_config={

            "automatic_activity_detection": {

                "disabled":
                    False,

                "prefix_padding_ms":
                    80,

                "silence_duration_ms":
                    350
            }
        },

        thinking_config={

            "thinking_level":
                "minimal"
        },

        input_audio_transcription={},

        output_audio_transcription={},

        speech_config={

            "voice_config": {

                "prebuilt_voice_config": {

                    "voice_name":
                        "Kore"
                }
            }
        },

        # Enable session resumption.
        session_resumption=
            types.SessionResumptionConfig(

                handle=
                    resume_handle
            )
    )

    if resume_handle:

        print(
            "\nNORA resuming session...",
            flush=True
        )

    else:

        print(
            "\nNORA connecting...",
            flush=True
        )

    try:

        async with client.aio.live.connect(

            model=
                GEMINI_MODEL,

            config=
                config

        ) as session:

            print()

            print(
                "NORA is ACTIVE.",
                flush=True
            )

            print(
                "NORA voice output enabled.",
                flush=True
            )

            print(
                "Speak your command.",
                flush=True
            )

            print()

            nora_is_speaking.clear()

            # ----------------------------------------------
            # IMPORTANT:
            # Keep microphone streaming and receiving
            # alive concurrently.
            # ----------------------------------------------

            sender_task = asyncio.create_task(
                audio_sender(
                    session
                )
            )

            receiver_task = asyncio.create_task(
                response_receiver(
                    session
                )
            )

            try:

                await asyncio.gather(
                    sender_task,
                    receiver_task
                )

            finally:

                sender_task.cancel()
                receiver_task.cancel()

                await asyncio.gather(
                    sender_task,
                    receiver_task,
                    return_exceptions=True
                )

    except asyncio.CancelledError:

        raise

    except Exception as error:

        if not (
            stop_program
            or exit_active_session.is_set()
        ):

            raise error

    finally:

        nora_is_speaking.clear()

        clear_input_audio()

        clear_output_audio()


# ============================================================
# ACTIVE NORA SESSION
# ============================================================

async def active_nora():

    global session_resume_handle
    global stop_program

    reconnect_count = 0

    # --------------------------------------------------------
    # Speaker output stream
    # --------------------------------------------------------

    speaker_stream = sd.OutputStream(

        samplerate=
            OUTPUT_SAMPLE_RATE,

        channels=
            CHANNELS,

        dtype=
            "int16",

        callback=
            output_callback
    )

    speaker_stream.start()

    try:

        while (
            not stop_program
            and not exit_active_session.is_set()
        ):

            try:

                clear_input_audio()
                clear_output_audio()

                nora_is_speaking.clear()

                # --------------------------------------------
                # Microphone
                # --------------------------------------------

                with sd.RawInputStream(

                    samplerate=
                        INPUT_SAMPLE_RATE,

                    blocksize=
                        LIVE_BLOCK_SIZE,

                    channels=
                        CHANNELS,

                    dtype=
                        "int16",

                    callback=
                        microphone_callback

                ):

                    await run_nora_live(
                        resume_handle=
                            session_resume_handle
                    )

                if (
                    stop_program
                    or exit_active_session.is_set()
                ):

                    break

                # The Live session ended unexpectedly.
                reconnect_count += 1

            except asyncio.CancelledError:

                break

            except Exception as error:

                if (
                    stop_program
                    or exit_active_session.is_set()
                ):

                    break

                reconnect_count += 1

                # ------------------------------------------------
                # Hide raw WebSocket error from normal output.
                # ------------------------------------------------

                print(
                    "\nNORA reconnecting...",
                    flush=True
                )

                # ------------------------------------------------
                # Exponential backoff
                # ------------------------------------------------

                wait_time = min(

                    2 ** reconnect_count,

                    MAX_RECONNECT_DELAY
                )

                print(
                    f"Retrying in "
                    f"{wait_time} seconds...",
                    flush=True
                )

                try:

                    await asyncio.sleep(
                        wait_time
                    )

                except asyncio.CancelledError:

                    break

                # ------------------------------------------------
                # After too many attempts, discard stale
                # session handle and start a clean session.
                # ------------------------------------------------

                if (
                    reconnect_count
                    >= MAX_RECONNECTS
                ):

                    print(
                        "Starting a fresh NORA session...",
                        flush=True
                    )

                    session_resume_handle = None

                    reconnect_count = 0

            else:

                # A cleanly ended session resets
                # the reconnect counter.
                reconnect_count = 0

    finally:

        nora_is_speaking.clear()

        try:

            speaker_stream.stop()

        except Exception:

            pass

        try:

            speaker_stream.close()

        except Exception:

            pass

        clear_input_audio()
        clear_output_audio()


# ============================================================
# MAIN
# ============================================================

def main():

    global stop_program

    print()

    print(
        "=" * 65
    )

    print(
        "              NORA HARDWARE AI"
    )

    print(
        "=" * 65
    )

    print()

    print(
        f"Gemini model: "
        f"{GEMINI_MODEL}"
    )

    # --------------------------------------------------------
    # ESP32
    # --------------------------------------------------------

    print(
        "Checking ESP32..."
    )

    if check_esp32():

        print(
            "ESP32 connection: OK"
        )

    else:

        print(
            "WARNING: ESP32 is not reachable."
        )

    # --------------------------------------------------------
    # VOSK
    # --------------------------------------------------------

    recognizer = load_vosk()

    print()

    print(
        "NORA is ready."
    )

    # --------------------------------------------------------
    # MAIN SLEEP/ACTIVE LOOP
    # --------------------------------------------------------

    try:

        while not stop_program:

            exit_active_session.clear()

            # -----------------------------------------------
            # Sleep mode
            # -----------------------------------------------

            activated = wait_for_nora(
                recognizer
            )

            if not activated:

                break

            # -----------------------------------------------
            # Active mode
            # -----------------------------------------------

            exit_active_session.clear()

            asyncio.run(
                active_nora()
            )

            if stop_program:

                break

            if exit_active_session.is_set():

                print()
                print(
                    "NORA returned to sleep mode.",
                    flush=True
                )

            else:

                print()
                print(
                    "NORA returning to wake mode...",
                    flush=True
                )

            print()

    except KeyboardInterrupt:

        stop_program = True

        stop_audio_devices()

    except Exception as error:

        print()
        print(
            "NORA error:"
        )
        print(
            error
        )

        stop_program = True

        stop_audio_devices()

    finally:

        stop_program = True

        nora_is_speaking.clear()

        stop_audio_devices()

        clear_input_audio()
        clear_output_audio()

        print()
        print(
            "NORA stopped.",
            flush=True
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        stop_program = True

        stop_audio_devices()

        print(
            "\nNORA stopped.",
            flush=True
        )

    except Exception as error:

        print(
            f"\nFatal error: {error}",
            flush=True
        )

        sys.exit(1)