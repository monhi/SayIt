// CKeywordSpotter.h

#pragma once

#include <onnxruntime_cxx_api.h>
#include "kissfft/kiss_fftr.h"
#include <deque>
#include <vector>
#include <string>
#include <array>
#include <memory>

constexpr int SAMPLE_RATE = 16000;
constexpr int DURATION = 1;                  // 1 second
constexpr int WINDOW_SIZE = SAMPLE_RATE * DURATION;  // 16000 samples
constexpr int N_FFT = 512;
constexpr int WIN_LENGTH = 400;              // 25 ms @ 16kHz
constexpr int HOP_LENGTH = 160;              // 10 ms @ 16kHz
constexpr int N_MELS = 40;
constexpr int FRAMES = 101;                  // Exact number of frames for 1-second audio

constexpr int SUPPRESSION_DURATION = 200;    // ~2 seconds (200 frames at 10ms hop)

constexpr const char* LABELS[] = {
    "down", "go", "left", "no",
    "right", "stop", "up", "yes",
    "_noise_"
};
constexpr int NUM_CLASSES = sizeof(LABELS) / sizeof(LABELS[0]);
constexpr int NOISE_INDEX = 8;

class KeywordSpotter {
public:
    /**
     * @brief Construct a new KeywordSpotter object
     * @param onnxPath Path to the exported ONNX model (kws.onnx)
     */
    explicit KeywordSpotter(const std::wstring& onnxPath);

    /**
     * @brief Destructor - cleans up FFT config and ONNX resources
     */
    ~KeywordSpotter();

    /**
     * @brief Process a chunk of incoming audio samples (real-time streaming)
     * @param samples New audio samples (16kHz, float, range usually [-1, 1])
     * @return true if at least one inference was performed
     */
    bool processAudio(const std::vector<float>& samples);

    /**
     * @brief Process an entire WAV file buffer at once (offline mode)
     * @param samples Full audio samples (16kHz, float)
     */
    void processWavFile(const std::vector<float>& samples);

private:
    // ONNX Runtime members
    Ort::Env m_env;
    Ort::Session m_session{ nullptr };
    std::string m_inputName;
    std::string m_outputName;

    // Audio buffer for streaming
    std::deque<float> m_audioBuffer;
    int m_samplesSinceLastInfer = 0;

    // Feature extraction
    kiss_fftr_cfg m_fftCfg;
    std::vector<float> m_fftFrame;              // size: N_FFT (padded)
    std::vector<kiss_fft_cpx> m_fftOut;          // size: N_FFT/2 + 1
    std::vector<float> m_hannWindow;             // size: WIN_LENGTH
    std::vector<std::vector<float>> m_melFilterBank;  // [N_MELS][N_FFT/2 + 1]
    std::vector<float> m_features;              // flattened: N_MELS * FRAMES

    // State machine for detection suppression
    std::string m_lastDetectedWord;
    int m_suppressionFrames = 0;

    // Initialization helpers
    void initHannWindow();
    void initMelFilterBank();
    float hz_to_mel(float hz);
    float mel_to_hz(float mel);

    // Processing steps
    void extractLogMel();
    void runInference();

    // Utility
    bool isSilent() const;
    static int argmax(const float* data, int size);
};