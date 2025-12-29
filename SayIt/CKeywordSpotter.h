#pragma once
#include <vector>
#include <deque>
#include <string>
#include <onnxruntime_cxx_api.h>

static constexpr int SAMPLE_RATE = 16000;
static constexpr int WINDOW_SIZE = 16000;
static constexpr int FRAME_LEN = 400;
static constexpr int HOP_LEN = 160;
static constexpr int FFT_SIZE = 512;
static constexpr int MEL_BINS = 40;
static constexpr int NUM_FRAMES = 98;
static constexpr int NUM_CLASSES = 9;

static const char* LABELS[NUM_CLASSES] =
{
    "down", "go", "left", "no",
    "right", "stop", "up", "yes",
    "_noise_"
};


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

private:
    // Audio buffer (1 second)
    std::deque<float> m_audioBuffer;

    // ONNX Runtime
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::MemoryInfo m_memoryInfo;

    // Feature buffer: [1, 1, 40, 98]
    std::vector<float> m_features;
    size_t m_samplesSinceLastInfer = 0;
    static constexpr size_t INFER_HOP = 1600; // 100 ms

};
