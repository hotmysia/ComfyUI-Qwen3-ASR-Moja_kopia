import torch
import numpy as np
import os
import torch.nn.functional as F
import folder_paths
from qwen_asr import Qwen3ASRModel

# Global cache for models to prevent reloading
_QWEN3_MODEL_CACHE = {}

def get_qwen_model_list():
    all_files = folder_paths.get_filename_list("diffusion_models")
    qwen_models = set()
    for f in all_files:
        # Look for the specific structure: Qwen3-ASR/ModelFolder/
        if "Qwen3-ASR" in f:
            f = f.replace("\\", "/")
            parts = f.split("/")
            if len(parts) >= 2:
                # We want the path up to the second part (e.g. Qwen3-ASR/Qwen3-ASR-1.7B)
                qwen_models.add("/".join(parts[:2]))
    
    res = sorted(list(qwen_models))
    return res if res else ["None Found (Place in models/diffusion_models/Qwen3-ASR/)"]

def resolve_qwen_path(model_name):
    # Try standard Comfy resolution first
    path = folder_paths.get_full_path("diffusion_models", model_name)
    if path:
        return os.path.dirname(path) if os.path.isfile(path) else path
    
    # Fallback: manual join with configured diffusion_models paths for directory support
    for base in folder_paths.get_folder_paths("diffusion_models"):
        full_path = os.path.join(base, model_name)
        if os.path.exists(full_path):
            return full_path
    return model_name

class Qwen3ForcedAlignerConfig:
    """
    Provides configuration for the Qwen3 Forced Aligner.
    This config is passed to the main ASR node to enable timestamp generation.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_qwen_model_list(),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("QWEN3_ALIGNER_CONF",)
    RETURN_NAMES = ("aligner_config",)
    FUNCTION = "get_config"
    CATEGORY = "Qwen3-ASR"

    def get_config(self, model_name, device, precision):
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        
        load_path = resolve_qwen_path(model_name)
        return ({
            "model_name": load_path,
            "kwargs": {
                "device_map": device,
                "dtype": dtype_map[precision]
            }
        },)

class Qwen3ASRTranscriber:
    """
    Performs ASR inference using Qwen3-ASR models.
    Optionally uses a Forced Aligner config to generate word-level timestamps.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (get_qwen_model_list(),),
                "language": ("STRING", {"default": "auto"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
            },
            "optional": {
                "forced_aligner": ("QWEN3_ALIGNER_CONF",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps")
    FUNCTION = "transcribe"
    CATEGORY = "Qwen3-ASR"

    def transcribe(self, audio, model_name, language, device, precision, max_new_tokens, forced_aligner=None):
        global _QWEN3_MODEL_CACHE

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        dtype = dtype_map[precision]
        lang_param = None if not language or language.lower() == "auto" else language

        # Create a unique cache key based on model settings
        aligner_name = forced_aligner["model_name"] if forced_aligner else "none"
        cache_key = f"{model_name}_{device}_{precision}_{aligner_name}"

        if cache_key not in _QWEN3_MODEL_CACHE:
            load_path = resolve_qwen_path(model_name)
            print(f"[Qwen3-ASR] Loading model from: {load_path}...")
            
            loader_kwargs = {
                "pretrained_model_name_or_path": load_path,
                "dtype": dtype,
                "device_map": device,
                "max_new_tokens": max_new_tokens,
            }

            if forced_aligner:
                loader_kwargs["forced_aligner"] = forced_aligner["model_name"]
                loader_kwargs["forced_aligner_kwargs"] = forced_aligner["kwargs"]

            _QWEN3_MODEL_CACHE[cache_key] = Qwen3ASRModel.from_pretrained(**loader_kwargs)

        model = _QWEN3_MODEL_CACHE[cache_key]

        # Prepare Audio
        # ComfyUI audio format: {"waveform": [Batch, Channels, Samples], "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Convert to mono if necessary and remove batch dim
        if waveform.ndim == 3:
            waveform = waveform[0]
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform[0]

        # Resample to 16000Hz using torch to avoid buggy librosa calls in Python 3.13
        if sample_rate != 16000:
            # F.interpolate expects [batch, channels, length]
            w = waveform.view(1, 1, -1)
            w = F.interpolate(w, size=int(w.shape[-1] * 16000 / sample_rate), mode='linear', align_corners=False)
            waveform = w.view(-1)
            sample_rate = 16000

        # Qwen-ASR expects (numpy_array, sample_rate)
        audio_input = (waveform.cpu().numpy(), sample_rate)

        # Run Inference
        results = model.transcribe(
            audio=[audio_input], # Wrap in list to match expected batch format
            language=lang_param,
            return_time_stamps=True if forced_aligner else False
        )

        res = results[0]
        transcription_text = res.text
        
        # Format Timestamps if available
        timestamp_output = ""
        if forced_aligner and hasattr(res, 'time_stamps') and res.time_stamps:
            ts_lines = []
            # The aligner returns a list of timestamp objects
            for ts in res.time_stamps:
                # Format: [start_s - end_s] text
                ts_lines.append(f"[{ts.start_time:.2f} - {ts.end_time:.2f}] {ts.text}")
            timestamp_output = "\n".join(ts_lines)
        else:
            timestamp_output = "No timestamps generated (Forced Aligner not provided or failed)."

        return (transcription_text, timestamp_output)

NODE_CLASS_MAPPINGS = {
    "Qwen3ASRTranscriber": Qwen3ASRTranscriber,
    "Qwen3ForcedAlignerConfig": Qwen3ForcedAlignerConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRTranscriber": "Qwen3 ASR Transcriber",
    "Qwen3ForcedAlignerConfig": "Qwen3 Forced Aligner Config"
}