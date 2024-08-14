import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch    #install torch from pytorch.org
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

app = Flask(__name__)

# Global variables to be used across requests
data_queue = Queue()
transcription = ['']
recorder = None
source = None
audio_model = None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global data_queue, transcription, recorder, source, audio_model

    if not audio_model:
        return jsonify({"error": "Model not loaded."}), 500

    # Process incoming audio file from the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    audio_file = request.files['file']
    audio_data = audio_file.read()

    data_queue.put(audio_data)
    
    # Record and transcribe
    phrase_time = datetime.utcnow()
    audio_data = b''.join(data_queue.queue)
    data_queue.queue.clear()
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    transcription.append(text)

    return jsonify({"transcription": transcription})

def load_model(args):
    global recorder, source, audio_model

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

@app.route('/initialize', methods=['POST'])
def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)

    args = parser.parse_args()

    # Load the model and microphone
    load_model(args)

    return jsonify({"status": "Model loaded."})

if __name__ == "__main__":
    app.run(host='localhost', port=5000)
