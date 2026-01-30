# Qwen3 ASR Transcriber

The **Qwen3 ASR Transcriber** is the primary inference node for the Qwen3-ASR model family. It converts speech from audio input into text and can optionally generate precise word-level timestamps when paired with a forced aligner.

## Parameters

- **audio**: The input audio stream (usually from a *Load Audio* node).
- **model_name**: The directory name of the Qwen3-ASR model (0.6B or 1.7B) located in `models/diffusion_models/Qwen3-ASR/`.
- **language**: The target language for transcription. Use `auto` to allow the model to automatically identify the spoken language.
- **device**: The hardware device to run the model on (`cuda` or `cpu`).
- **precision**: The floating-point precision. `bf16` is recommended for modern NVIDIA GPUs to save VRAM without losing accuracy.
- **max_new_tokens**: The maximum number of tokens to generate in the output text. Increase this for longer audio files.
- **flash_attention_2**: Enable Flash Attention 2 for faster inference and lower VRAM usage. Requires a compatible NVIDIA GPU and the `flash-attn` package installed.
- **chunk_size**: Process audio in chunks of this many seconds (default: 30). Set to 0 to disable. This is critical for transcribing long audio files to prevent the model from exceeding its context window.
- **overlap**: The number of seconds of overlap between chunks (default: 2). This helps maintain context and prevents words from being cut off at chunk boundaries.
- **forced_aligner** (Optional): An optional input from the **Qwen3 Forced Aligner Config** node. If connected, the node will calculate and output word-level timestamps.

## Outputs

- **text**: The raw transcription of the audio.
- **timestamps**: A formatted string containing word-level timestamps (e.g., `[0.00 - 0.50] Hello`). If no aligner is provided, this will return a status message.

## Usage Tips

- **Resampling**: This node automatically resamples audio to 16kHz internally using Torch, ensuring compatibility with the model even if your input audio is 44.1kHz or 48kHz.
- **Long Audio**: For recordings longer than 30 seconds, ensure `chunk_size` is enabled. If you notice repetition or hallucinations at the end of long files, try adjusting the `chunk_size` or `overlap`.
- **Caching**: The model is cached in memory after the first run. Changing the `model_name`, `device`, or `precision` will trigger a reload.

## Example

1. Connect a **Load Audio** node to the **audio** input.
2. Select `Qwen3-ASR-1.7B` in **model_name**.
3. (Optional) Connect a **Qwen3 Forced Aligner Config** to the **forced_aligner** input to see timing data in the second output.