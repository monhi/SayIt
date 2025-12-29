#include "CKeywordSpotter.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include "kissfft/kiss_fftr.h"



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
    //m_samplesSinceLastInfer = 0;
    //INFER_HOP = 1600; // 100 ms

}

// =========================
// Public Audio Entry Point
// =========================
/*
bool CKeywordSpotter::processAudio(const std::vector<float>& samples)
{
    for (float s : samples)
        m_audioBuffer.push_back(s);

    if (m_audioBuffer.size() < WINDOW_SIZE)
        return false;

    while (m_audioBuffer.size() > WINDOW_SIZE)
        m_audioBuffer.pop_front();

    extractLogMel();
    runInference();

    return true;
}
*/
bool CKeywordSpotter::processAudio(const std::vector<float>& samples)
{
    for (float s : samples)
    {
        m_audioBuffer.push_back(s);
        m_samplesSinceLastInfer++;
    }

    if (m_audioBuffer.size() < WINDOW_SIZE)
        return false;

    while (m_audioBuffer.size() > WINDOW_SIZE)
        m_audioBuffer.pop_front();

    if (m_samplesSinceLastInfer < INFER_HOP)
        return false;

    m_samplesSinceLastInfer = 0;

    extractLogMel();
    runInference();

    return true;
}

// =========================
// Log-Mel Extraction (kissFFT)
// =========================

void CKeywordSpotter::extractLogMel()
{
    std::fill(m_features.begin(), m_features.end(), 0.0f);

    kiss_fftr_cfg fftCfg = kiss_fftr_alloc(FFT_SIZE, 0, nullptr, nullptr);

    std::vector<float> frame(FRAME_LEN);
    std::vector<float> window(FRAME_LEN);
    std::vector<kiss_fft_cpx> fftOut(FFT_SIZE / 2 + 1);

    // Hann window
    for (int i = 0; i < FRAME_LEN; i++)
        window[i] = 0.5f - 0.5f * cosf(2.0f * 3.1415926535f * i / FRAME_LEN);

    int frameIdx = 0;

    for (int offset = 0;
        offset + FRAME_LEN <= WINDOW_SIZE;
        offset += HOP_LEN)
    {
        for (int i = 0; i < FRAME_LEN; i++)
            frame[i] = m_audioBuffer[offset + i] * window[i];

        kiss_fftr(fftCfg, frame.data(), fftOut.data());

        for (int m = 0; m < MEL_BINS; m++)
        {
            float energy = 0.0f;

            // Simplified band energy (placeholder)
            for (int k = 1; k < FFT_SIZE / 2; k++)
                energy += fftOut[k].r * fftOut[k].r;

            m_features[m * NUM_FRAMES + frameIdx] =
                logf(energy + 1e-6f);
        }

        frameIdx++;
        if (frameIdx >= NUM_FRAMES)
            break;
    }

    free(fftCfg);
}

// =========================
// ONNX Inference
// =========================

void CKeywordSpotter::runInference()
{
    std::array<int64_t, 4> shape =
    {
        1, 1, MEL_BINS, NUM_FRAMES
    };

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

    std::cout << "idx:" << idx << std::endl;

    if (idx != NUM_CLASSES - 1 && conf > 0.7f)
    {
        std::cout << "Detected: "
            << LABELS[idx]
            << " ("
            << conf
            << ")"
            << std::endl;
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
