# parakeet-ultra-sherpa-onnx-int8

**Mirror of [`tpmaerk/parakeet-ultra-sherpa-onnx-int8`](https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8)** (Hugging Face).

An int8 [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) `nemo_transducer`
(TDT) bundle of **[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)** —
Moondream's retrain of NVIDIA's `parakeet-tdt-0.6b-v3` (FastConformer 24×1024, TDT
decoder, 25 European languages). Format conversion only; no further training.

⚠️ The conversion was done **by [tpmaerk](https://huggingface.co/tpmaerk)** — this GitHub
repo is a byte-identical mirror with the checksums, license files, and a load test added.
The original model card is preserved verbatim in [SOURCE-CARD.md](SOURCE-CARD.md).

It is a drop-in replacement for the official
[`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8)
bundle: same 4-file layout, same `model_type="nemo_transducer"` usage.

## Files

| file | bytes | sha256 |
|---|---|---|
| `encoder.int8.onnx` | 652282347 | `f4d06712d32431b1718eceba2153c314ac4425e8fe708a3534f5e56c5ad53697` |
| `decoder.int8.onnx` | 11845275 | `94373c6405c8149d1c9d1e9862514e0e3e88dab1c7de18fac9d5df9c6ba5efdd` |
| `joiner.int8.onnx` | 6355277 | `8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2` |
| `tokens.txt` | 93939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |

Checksums match the values published in the source repo and were re-verified after
download on this mirror. To re-verify locally:

```bash
git lfs pull
shasum -a 256 -c checksums.txt
```

## Usage (sherpa-onnx, requires ≥ 1.13 to load TDT models)

```python
import sherpa_onnx, soundfile as sf

rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx", decoder="decoder.int8.onnx", joiner="joiner.int8.onnx",
    tokens="tokens.txt", model_type="nemo_transducer", feature_dim=128,
    decoding_method="greedy_search", num_threads=4,
)
audio, sr = sf.read("audio.wav")
s = rec.create_stream()
s.accept_waveform(sr, audio)
rec.decode_stream(s)
print(s.result.text)
```

## Verified on this mirror (2026-10-08)

- **sha256**: all four files on this mirror hash to the checksums published in the
  source repo (table above) — byte-identical copy.
- **Load + decode test**, sherpa-onnx 1.13.8, macOS arm64 (Apple M1), 4 threads:
  - `bcn_weather.wav` (11.0 s) → *"Yesterday it was 35 degrees in Barcelona, but today
    the temperature will go down to minus 20 degrees."* — 26× realtime
  - `en-Alice_woman.wav` (9.3 s) → *"So just to clarify, we've had 19 to 20 year olds
    on our podcast so far. And if you don't mind me asking, how old are you?"* — 31× realtime
- These are short-clip smoke numbers on an M1, **not** a re-run of the benchmark below.

## Accuracy / speed reported by the converter (source-repo figures)

FLEURS `de_de` test split (862 clips, 189 min), CPU, 4 threads, greedy search,
Apple M4 Pro — reproduced verbatim from the source repo:

| Bundle | WER | Realtime factor |
|---|---|---|
| Parakeet V3 int8 (csukuangfj) | 6.82 % | 44.5× |
| **Parakeet Ultra int8 (this model)** | **5.42 %** | **46.2×** |

## License & attribution

**CC BY 4.0** — give credit to:

- **NVIDIA Corporation** — `parakeet-tdt-0.6b-v3` base model (CC BY 4.0)
- **Moondream** — `parakeet-ultra` fine-tune (CC BY 4.0)
- **tpmaerk (Tim-Patrick Märk)** — sherpa-onnx int8 conversion and measurements

See [LICENSE](LICENSE) (verbatim CC BY 4.0) and [NOTICE](NOTICE). Source of truth:
https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8
