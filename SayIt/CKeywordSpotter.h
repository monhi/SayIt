#pragma once

#include <vector>
#include <deque>
#include <string>
#include <array>
#include <onnxruntime_cxx_api.h>
#include "kissfft/kiss_fftr.h"

class CKeywordSpotter
{
public:
    explicit CKeywordSpotter(const std::wstring& onnxModelPath);

    // Feed microphone samples (float, 16kHz)
    // Returns true if inference was executed
    bool processAudio(const std::vector<float>& samples);

private:
    void extractLogMel();
    void runInference();
    void softmax(float* x, int n);
    int argmax(const float* x, int n);
    void initMelFilterBank();

private:
    // Audio buffer (1 second)
    std::deque<float> m_audioBuffer;

    // ONNX Runtime
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::MemoryInfo m_memoryInfo;

    // Feature buffer: [1, 1, 40, 98]
    std::vector<float> m_features;

    // Mel filter bank: 40x257
    std::vector<std::vector<float>> m_melFilterBank;

    // FFT config
    kiss_fftr_cfg m_fftCfg = nullptr;

    // Hop counter
    size_t m_samplesSinceLastInfer = 0;
    int m_lastDetectedWord = -1;

    // Constants
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int WINDOW_SIZE = 16000;
    static constexpr int FRAME_LEN = 400;
    static constexpr int HOP_LEN = 160;
    static constexpr int FFT_SIZE = 512;
    static constexpr int MEL_BINS = 40;
    static constexpr int NUM_FRAMES = 98;
    static constexpr int NUM_CLASSES = 9;

    static const char* LABELS[NUM_CLASSES];
};
