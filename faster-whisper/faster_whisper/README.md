You can simply edit the Faster-Whisper utils.py if you want to add models under custom names.
For example here, "quantized" does not exists in the base file.
This is located in the directory where you installed Faster-Whisper with git clone (./faster-whisper/faster_whisper/utils.py).
You can also load your models from a local dir using : model = faster_whisper.WhisperModel("whisper-large-v3-ct2")
instructions are on their github repo.
