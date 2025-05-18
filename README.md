# <u>**t2yLLM : a fast LLM Voice Assistant**</u>


## <u>🔥 Backends :</u>

- **vLLM** : really fast and well documented inference pipeline for your favorite LLM
- **Faster-Whisper** : incredibly fast STT for realtime text generation
- **piperTTS** : fast text to speech generating natural voice, maybe the best for french atm
- **Silero-vad** : process the audio buffer and prevents whisper hallucinations
- **pvporcupine** : keyword detection
- **Chromadb** : a vector search database that serves as the model memory
- **default LLM** : Qwen3 14B or other variants, GPTQ 4Bit quantized
- **Pytorch**

## <u>💡 Functionalities :</u>

- **t2yLLM** lets you speak to your device of choice (here a raspberry Pi with a respeaker hat from seeed studio)
and get an audio answer from your favorite LLM (here Qwen3 by default).
It should just work like any home assistant. The default keyword to activate speech detection is **"Ok Mars"**
but you can change it of course. ATM if you want a custom keyword, it is **mandatory** to create a Picovoice account (check [Picovoice](https://picovoice.ai/)), train and download a custom keyword to get a working pipeline.
- **Meteo** : It can search for meteo infos using your OpenWeather API key
- **Pokemon** : Look for any Pokemon info using Tyradex API (french)
- **Wikipedia** : Make Wikipedia searches using the python API
- **Vector Search** : stores all in a synthetic way in chromadb if needed and can retrieve the memorized info
- **t2yLLM** is meant to work on a 16GB GPU, but in order to achieve that, first launch the LLM backend script in order to avoid OOM

## <u>💡 Specifics :</u>

### - material :
  - **client side** :
    - Raspberry Pi 5, 4GB
    - respeaker like from seeed studio
  - **server side** :
    - An Nvidia GPU with 16GB of VRAM at least

### - Pipeline :
  - **t2yLLM** uses AsyncLLMEngine in order to stream tokens and generate sound from them as soon as possible.
  - The audio dispatcher processes text received from the LLM and transforms it to .flac segments and then
    sends them to the client (raspberry Pi)
  - Sound reveived from the Pi 5 is analyzed by silerovad to detect speech in addition to pvporcupine
  - Relevant sound is then translated by Faster-Whisper with low latency
  - The audio dispatcher transforms the LLM answer to speech with coqui TTS and then sends audio parts in .flac
    over the network to reduce bandwidth usage and decrease latency

## <u>⚙️ Parameters & specifics:</u>

- configuration should be done via the .yaml config file without having to directly interact with the code
- configuration can be enhanced via the YamlConfig_Loader.py
- **t2yLLM** should be used on local network only since all is in clear text for now
- the directory structure should be /t2yLLM/config and /t2yLLM/Chat

## <u>⚙️ Environment variables :</u>

- export PORCUPINE_KEY='myporcupinekey'
- export OPENWEATHERMAP_API_KEY='myopenweatherkey'
- export VLLM_ATTENTION_BACKEND=FLASH_ATTN #for V1 engine
- export VLLM_FLASH_ATTN_VERSION="2"       #for V1 engine
- export VLLM_USE_V1=1
- VLLM_WORKER_MULTIPROC_METHOD="spawn" #for V1 engine
- export TORCH_CUDA_ARCH_LIST='myarchitecture' #if needed



## <u>🔍 Github repositories used in order to make this code : </u>

- 🔗 [vLLM](https://github.com/vllm-project/vllm)
- 🔗 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- 🔗 [Silero-vad](https://github.com/snakers4/silero-vad)
- 🔗 [pvporcupine](https://github.com/Picovoice/porcupine)
- 🔗 [Whisper-streaming](https://github.com/ufal/whisper_streaming)
- 🔗 [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- 🔗 [Chromadb](https://github.com/chroma-core/chroma)
- 🔗 [piperTTS](https://github.com/rhasspy/piper)
- 🔗 [json_repair](https://github.com/mangiucugna/json_repair)

## <u>🔍 APIs :</u>

- 🔗 [Tyradex](https://tyradex.vercel.app/)
- 🔗 [OpenWeather](https://openweathermap.org/)

## <u>🛠️ ToDo : </u>

- Keep context between interactions
- improve processing pipeline
- Switch from UDP to Quic
- switch or add option to use OpenWakeWord but I could not make it train on a custom wake word

## <u>⚖️ License :</u>

This code is under the **MIT** license. Please mention me as the author if you found this code useful
  [![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

Copyright (c) 2025 Saga9103

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
