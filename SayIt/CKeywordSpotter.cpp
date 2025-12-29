#include "CKeywordSpotter.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "kissfft/kiss_fftr.h"
#define M_PI       3.14159265358979323846   // pi


const char* CKeywordSpotter::LABELS[NUM_CLASSES] =
{
    "down", "go", "left", "no",
    "right", "stop", "up", "yes",
    "_noise_"
};

// =========================
// Constructor
// =========================
CKeywordSpotter::CKeywordSpotter(const std::wstring& onnxModelPath)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "kws"),
    m_session(nullptr),
    m_memoryInfo(Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    m_session = Ort::Session(m_env, onnxModelPath.c_str(), opts);

    m_features.resize(1 * 1 * MEL_BINS * NUM_FRAMES);

    m_fftCfg = kiss_fftr_alloc(FFT_SIZE, 0, nullptr, nullptr);

    initMelFilterBank();
}

// =========================
// Mel filter bank initialization
// =========================
void CKeywordSpotter::initMelFilterBank()
{
    m_melFilterBank.resize(MEL_BINS, std::vector<float>(FFT_SIZE / 2 + 1, 0.0f));

    auto hzToMel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto melToHz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

    float melLow = hzToMel(0);
    float melHigh = hzToMel(SAMPLE_RATE / 2);
    std::vector<float> melPoints(MEL_BINS + 2);
    for (int i = 0; i < MEL_BINS + 2; i++)
        melPoints[i] = melToHz(melLow + (melHigh - melLow) * i / (MEL_BINS + 1));

    // Bin indices
    std::vector<int> bin(MEL_BINS + 2);
    for (int i = 0; i < MEL_BINS + 2; i++)
        bin[i] = static_cast<int>(floorf((FFT_SIZE + 1) * melPoints[i] / SAMPLE_RATE));

    for (int m = 0; m < MEL_BINS; m++)
    {
        for (int k = bin[m]; k < bin[m + 1]; k++)
            m_melFilterBank[m][k] = (k - bin[m]) / float(bin[m + 1] - bin[m]);
        for (int k = bin[m + 1]; k < bin[m + 2]; k++)
            m_melFilterBank[m][k] = (bin[m + 2] - k) / float(bin[m + 2] - bin[m + 1]);
    }
}

// =========================
// Process audio
// =========================
bool CKeywordSpotter::processAudio(const std::vector<float>& samples)
{
    for (float s : samples)
        m_audioBuffer.push_back(s);

    while (m_audioBuffer.size() > WINDOW_SIZE)
        m_audioBuffer.pop_front();

    m_samplesSinceLastInfer += samples.size();
    if (m_audioBuffer.size() < WINDOW_SIZE || m_samplesSinceLastInfer < HOP_LEN * 10) // 1600 samples hop
        return false;

    m_samplesSinceLastInfer = 0;

    extractLogMel();
    runInference();

    return true;
}

// =========================
// Extract log-mel
// =========================
void CKeywordSpotter::extractLogMel()
{
    std::fill(m_features.begin(), m_features.end(), 0.0f);

    std::vector<float> frame(FRAME_LEN);
    std::vector<float> window(FRAME_LEN);

    for (int i = 0; i < FRAME_LEN; i++)
        window[i] = 0.5f - 0.5f * cosf(2 * M_PI * i / FRAME_LEN);

    kiss_fft_cpx fftOut[FFT_SIZE / 2 + 1];

    for (int frame_idx = 0; frame_idx < NUM_FRAMES; frame_idx++)
    {
        int offset = frame_idx * HOP_LEN;
        for (int i = 0; i < FRAME_LEN; i++)
            frame[i] = m_audioBuffer[offset + i] * window[i];

        kiss_fftr(m_fftCfg, frame.data(), fftOut);

        for (int m = 0; m < MEL_BINS; m++)
        {
            float energy = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; k++)
            {
                float mag = sqrtf(fftOut[k].r * fftOut[k].r + fftOut[k].i * fftOut[k].i);
                energy += m_melFilterBank[m][k] * mag * mag;
            }
            m_features[m * NUM_FRAMES + frame_idx] = logf(energy + 1e-6f);
        }
    }
}

// =========================
// Run ONNX inference
// =========================
void CKeywordSpotter::runInference()
{
    std::array<int64_t, 4> shape = { 1,1,MEL_BINS,NUM_FRAMES };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        m_features.data(),
        m_features.size(),
        shape.data(),
        shape.size()
    );

    const char* inputNames[] = { "input" };
    const char* outputNames[] = { "logits" };

    auto outputs = m_session.Run(
        Ort::RunOptions{ nullptr },
        inputNames, &inputTensor, 1,
        outputNames, 1
    );

    float* logits = outputs[0].GetTensorMutableData<float>();
    softmax(logits, NUM_CLASSES);

    int idx = argmax(logits, NUM_CLASSES);
    float conf = logits[idx];

    if (idx != NUM_CLASSES - 1 && conf > 0.7f && idx != m_lastDetectedWord)
    {
        std::cout << "Detected: " << LABELS[idx] << " (" << conf << ")" << std::endl;
        m_lastDetectedWord = idx;
    }
}

// =========================
// Utilities
// =========================
void CKeywordSpotter::softmax(float* x, int n)
{
    float maxVal = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

int CKeywordSpotter::argmax(const float* x, int n)
{
    return std::max_element(x, x + n) - x;
}
