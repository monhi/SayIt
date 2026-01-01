import sys
import numpy as np
import librosa
import onnxruntime as ort

# =========================
# Configuration (must match training)
# =========================
SAMPLE_RATE = 16000
DURATION = 1.0           # 1 second
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
NUM_FRAMES = 98

LABELS = [
    "down", "go", "left", "no",
    "right", "stop", "up", "yes",
    "_noise_"
]

# =========================
# Feature extraction
# =========================
def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    # Pad or truncate to exactly 1 second
    expected_len = int(SAMPLE_RATE * DURATION)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False  # Important to avoid extra padding
    )

    log_mel = np.log(mel + 1e-6)

    # Ensure exactly NUM_FRAMES frames
    if log_mel.shape[1] < NUM_FRAMES:
        pad_width = NUM_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad_width)), mode='constant')
    elif log_mel.shape[1] > NUM_FRAMES:
        log_mel = log_mel[:, :NUM_FRAMES]

    return log_mel.astype(np.float32)

# =========================
# Softmax
# =========================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =========================
# Main function
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kws.py <audio_file.wav>")
        return

    wav_file = sys.argv[1]

    features = extract_features(wav_file)
    # reshape to [1,1,n_mels, num_frames]
    input_tensor = features[np.newaxis, np.newaxis, :, :]

    # Load ONNX model
    session = ort.InferenceSession("kws.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for i in session.get_inputs():
        print(i.name, i.shape)
    for o in session.get_outputs():
        print(o.name, o.shape)
    

    outputs = session.run(
        [output_name],
        {input_name: input_tensor}
    )

    logits = outputs[0][0]  # shape [num_classes]
    probs = softmax(logits)
    idx = np.argmax(probs)
    print(f"Predicted keyword: {LABELS[idx]} ({probs[idx]:.2f})")

if __name__ == "__main__":
    main()
