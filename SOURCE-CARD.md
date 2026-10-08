---
license: cc-by-4.0
base_model: moondream/parakeet-ultra
library_name: sherpa-onnx
pipeline_tag: automatic-speech-recognition
tags:
  - parakeet
  - nemo
  - tdt
  - sherpa-onnx
  - onnx
  - int8
language:
  - de
  - en
  - multilingual
---

# Parakeet Ultra — sherpa-onnx int8

[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) converted to the
sherpa-onnx `nemo_transducer` format (int8), as a drop-in replacement for
[csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8).
Used by Meeting Miner as a local batch transcription model.

## Origin and license

- Base model: **NVIDIA Parakeet TDT 0.6B v3** ([nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)), CC BY 4.0.
- Fine-tune: **Parakeet Ultra by Moondream** ([moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)), CC BY 4.0.
- This repository: format conversion only, no further training. Licensed CC BY 4.0.

## Changes made

1. The Hugging Face checkpoint (`model.safetensors`, revision `73175eb7`) was mapped back onto the
   NeMo state dict of `nvidia/parakeet-tdt-0.6b-v3` (revision `541d1f99`) by key renaming only
   (inverse of `convert_nemo_to_hf.py` in `transformers`). The six `vad_head.*` tensors of Ultra are
   not part of the NeMo architecture and were dropped. Weights stored as F16 in the source were cast
   to F32.
2. Exported with `export_onnx.py` from
   [k2-fsa/sherpa-onnx `scripts/nemo/parakeet-tdt-0.6b-v3`](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3)
   (int8 dynamic quantization). The encoder metadata `url` contains "tdt"
   (`https://huggingface.co/moondream/parakeet-ultra (parakeet-tdt-0.6b-v3 architecture)`) —
   sherpa-onnx 1.13.x detects TDT models by that substring and refuses to load the model otherwise.

Checks performed: the same mapping applied to NVIDIA's own HF checkpoint reproduces all 723 tensors
of the original `.nemo` bit-exactly; the resulting Ultra `.nemo` is bit-identical to an independent
conversion ([Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx));
two exports produce byte-identical files; the same export chain applied to V3 reproduces the
csukuangfj bundle's transcripts on all 862 FLEURS German test clips. `tokens.txt` is identical to V3.

## Measurement (own, not the model card)

sherpa-onnx 1.13.8, CPU, 4 threads, greedy search, Apple M4 Pro, FLEURS de_de test split
(862 clips, 189 min), simple normalization (lower case, punctuation removed):

| Bundle | WER | Realtime factor |
|---|---|---|
| Parakeet V3 int8 (csukuangfj) | 6.82 % | 44.5× |
| **Parakeet Ultra int8 (this repo)** | **5.42 %** | **46.2×** |

## Files

| File | Bytes | sha256 |
|---|---|---|
| encoder.int8.onnx | 652282347 | f4d06712d32431b1718eceba2153c314ac4425e8fe708a3534f5e56c5ad53697 |
| decoder.int8.onnx | 11845275 | 94373c6405c8149d1c9d1e9862514e0e3e88dab1c7de18fac9d5df9c6ba5efdd |
| joiner.int8.onnx | 6355277 | 8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2 |
| tokens.txt | 93939 | d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d |

## Usage (sherpa-onnx)

```python
import sherpa_onnx
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx", decoder="decoder.int8.onnx", joiner="joiner.int8.onnx",
    tokens="tokens.txt", model_type="nemo_transducer", feature_dim=128,
)
```
