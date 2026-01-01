# SayIt - Lightweight Keyword Spotting in C++

**SayIt** is a real-time keyword spotting (KWS) application that detects a small set of voice commands on Windows using a tiny convolutional neural network.

### Supported Keywords
The system can reliably detect the following 8 commands (plus background noise):

- `down`
- `go`
- `left`
- `no`
- `right`
- `stop`
- `up`
- `yes`

A special `_noise_` class helps reject non-keyword audio and silence.

### Features
- Training in Python using PyTorch → exported to ONNX
- Real-time inference in C++ using ONNX Runtime
- Audio feature extraction implemented from scratch in C++ (no external dependencies except KissFFT and ONNX Runtime)
- Exact replication of librosa mel-spectrogram pipeline:
  - 16 kHz mono audio
  - 25 ms Hann window (400 samples)
  - 10 ms hop (160 samples)
  - 512-point FFT
  - 40 mel filterbanks (Slaney-style normalization)
  - Reflective padding
  - Log mel energies (no additional normalization)
- Simple energy-based Voice Activity Detection (VAD) to reduce false triggers during silence
- Streaming microphone input support
- Offline WAV file processing mode

### Tech Stack
- **Training**: Python 3, PyTorch, librosa, NumPy
- **Inference**: C++17, ONNX Runtime (CPU), KissFFT
- **FFT**: KissFFT (single-precision real FFT)
- **Mel filterbank**: Custom implementation matching librosa behavior
- **Build**: Visual Studio 2022 (MSVC) – recommended and default development environment

### Project Structure
SayIt/
├── train_kws.py              # Training script (PyTorch → ONNX export)
├── kws.onnx                  # Trained model (must be placed next to exe)
├── kws.onnx.data             # External data file for some ONNX models (if present)
├── onnxruntime.dll           # Required runtime DLL (copy next to exe)
├── KeywordSpotter.cpp        # Core inference + feature extraction
├── CKeywordSpotter.h         # Header with all constants and class definition
├── main.cpp                  # Example microphone / WAV processing loop
├── kissfft/                  # KissFFT library (submodule or included)
└── README.md


### Build Instructions (Windows)
1. Open the solution in **Visual Studio 2022**.
2. Ensure the project is configured for **x64** (recommended).
3. Include paths:
   - ONNX Runtime headers
   - KissFFT headers
4. Link against:
   - `onnxruntime.lib`
5. Build → generates `SayIt.exe`

### Runtime Requirements
After building, copy the following files into the same folder as `SayIt.exe`:
- `kws.onnx` ← **required** (produced by training script)
- `kws.onnx.data` ← **if present** (some ONNX exports split weights)
- `onnxruntime.dll` ← **required** (from ONNX Runtime Windows package)

Without these files next to the executable, the program will fail to load the model.

### Training the Model
1. Prepare dataset in `data/` folder with subfolders named exactly as the keywords:
Each folder should contain 16 kHz mono WAV files (1 second or less).

2. Run:

python train_kws.py


This trains a small CNN, validates performance, and exports kws.onnx.

Copy the newly generated kws.onnx (and kws.onnx.data if created) next to your compiled executable.

### Usage

Microphone mode (default): Speak one of the keywords → detection printed in console.
WAV file mode: Modify main.cpp to load and process a WAV file.

False positives during complete silence are minimized using energy-based VAD and noise-class thresholding.
Performance

Model size: ~50-100 KB
Inference time: < 5 ms per 1-second frame on modern CPU
Very low memory footprint
Suitable for embedded or always-on applications

### Credits & Libraries

ONNX Runtime
KissFFT
Training inspired by Google Speech Commands dataset pipeline

### License
MIT License – feel free to use, modify, and distribute.

Enjoy your always-listening lightweight keyword spotter!