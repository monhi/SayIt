#include "CKeywordSpotter.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>
#include <deque>

KeywordSpotter::KeywordSpotter(const std::wstring& onnxPath)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "kws")
    , m_session(m_env, onnxPath.c_str(), Ort::SessionOptions{ nullptr })
    , m_fftCfg(nullptr)
{
    // Get input/output names
    auto allocator = Ort::AllocatorWithDefaultOptions();
    m_inputName = m_session.GetInputNameAllocated(0, allocator).get();
    m_outputName = m_session.GetOutputNameAllocated(0, allocator).get();

    // Allocate FFT
    m_fftCfg = kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr);
    if (!m_fftCfg) {
        throw std::runtime_error("Failed to allocate KissFFT config");
    }

    m_fftFrame.resize(N_FFT, 0.0f);
    m_fftOut.resize(N_FFT / 2 + 1);

    // Initialize windows and filters
    initHannWindow();
    initMelFilterBank();

    // Feature buffer: N_MELS × FRAMES
    m_features.resize(N_MELS * FRAMES, 0.0f);
}

KeywordSpotter::~KeywordSpotter()
{
    if (m_fftCfg) {
        kiss_fft_free(m_fftCfg);
    }
}

void KeywordSpotter::initHannWindow()
{
    m_hannWindow.resize(WIN_LENGTH);
    const float PI = 3.14159265358979323846f;
    for (int i = 0; i < WIN_LENGTH; ++i) {
        m_hannWindow[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / (WIN_LENGTH - 1));
    }
}

float KeywordSpotter::hz_to_mel(float hz)
{
    // Slaney-style mel scale (matches modern librosa)
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (hz < min_log_hz) {
        return (hz - f_min) / f_sp;
    }
    else {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
}

float KeywordSpotter::mel_to_hz(float mel)
{
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (mel < min_log_mel) {
        return f_min + f_sp * mel;
    }
    else {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
}

void KeywordSpotter::initMelFilterBank()
{
    m_melFilterBank.assign(N_MELS, std::vector<float>(N_FFT / 2 + 1, 0.0f));

    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(SAMPLE_RATE / 2.0f);

    std::vector<float> mel_points(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (N_MELS + 1.0f);
    }

    std::vector<float> hz_points(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Build triangular filters
    for (int m = 0; m < N_MELS; ++m) {
        float left = hz_points[m];
        float center = hz_points[m + 1];
        float right = hz_points[m + 2];

        for (int k = 0; k <= N_FFT / 2; ++k) {
            float freq = static_cast<float>(k) * SAMPLE_RATE / N_FFT;
            if (freq >= left && freq <= center) {
                m_melFilterBank[m][k] = (freq - left) / (center - left);
            }
            else if (freq > center && freq <= right) {
                m_melFilterBank[m][k] = (right - freq) / (right - center);
            }
        }
    }

    // Apply Slaney normalization: 2 / (right - left)
    for (int m = 0; m < N_MELS; ++m) {
        float enorm = 2.0f / (hz_points[m + 2] - hz_points[m]);
        for (int k = 0; k <= N_FFT / 2; ++k) {
            m_melFilterBank[m][k] *= enorm;
        }
    }
}

bool KeywordSpotter::processAudio(const std::vector<float>& samples)
{
    for (float s : samples) {
        m_audioBuffer.push_back(s);
    }

    if (m_audioBuffer.size() < WINDOW_SIZE) {
        return false;
    }

    // Simple hop control (process every HOP_LENGTH new samples)
    m_samplesSinceLastInfer += samples.size();
    if (m_samplesSinceLastInfer < HOP_LENGTH) {
        return false;
    }
    m_samplesSinceLastInfer = 0;

    while (m_audioBuffer.size() >= WINDOW_SIZE) {
        extractLogMel();
        runInference();

        // Slide window
        for (int i = 0; i < HOP_LENGTH; ++i) {
            m_audioBuffer.pop_front();
        }
    }

    return true;
}

void KeywordSpotter::processWavFile(const std::vector<float>& samples)
{
    m_audioBuffer.clear();
    for (float s : samples) {
        m_audioBuffer.push_back(s);
    }

    if (m_audioBuffer.size() < WINDOW_SIZE) {
        std::cout << "WAV too short\n";
        return;
    }

    size_t offset = 0;
    while (offset + WINDOW_SIZE <= m_audioBuffer.size()) {
        // Copy current 1-second window
        std::deque<float> window(m_audioBuffer.begin() + offset,
            m_audioBuffer.begin() + offset + WINDOW_SIZE);
        m_audioBuffer = std::move(window);

        extractLogMel();
        runInference();

        offset += HOP_LENGTH;
    }
}

void KeywordSpotter::extractLogMel()
{
    assert(m_audioBuffer.size() >= WINDOW_SIZE);

    // Reflective padding to match librosa's default 'reflect' mode
    const int pad = N_FFT / 2; // 256
    std::vector<float> padded(WINDOW_SIZE + 2 * pad);

    // Left reflect
    for (int i = 0; i < pad; ++i) {
        padded[i] = m_audioBuffer[pad - 1 - i];
    }
    // Center
    std::copy(m_audioBuffer.begin(), m_audioBuffer.end(), padded.begin() + pad);
    // Right reflect
    for (int i = 0; i < pad; ++i) {
        padded[pad + WINDOW_SIZE + i] = m_audioBuffer[WINDOW_SIZE - 2 - i];
    }

    for (int frame_idx = 0; frame_idx < FRAMES; ++frame_idx) {
        int start = frame_idx * HOP_LENGTH;

        // Apply window
        for (int i = 0; i < WIN_LENGTH; ++i) {
            m_fftFrame[i] = padded[start + i] * m_hannWindow[i];
        }
        // Zero-pad rest
        std::fill(m_fftFrame.begin() + WIN_LENGTH, m_fftFrame.end(), 0.0f);

        // FFT
        kiss_fftr(m_fftCfg, m_fftFrame.data(), m_fftOut.data());

        // Mel energy
        for (int m = 0; m < N_MELS; ++m) {
            float energy = 0.0f;
            for (int k = 0; k <= N_FFT / 2; ++k) {
                float re = m_fftOut[k].r;
                float im = m_fftOut[k].i;
                energy += m_melFilterBank[m][k] * (re * re + im * im);
            }
            energy = std::max(energy, 1e-10f);
            m_features[m * FRAMES + frame_idx] = std::log(energy);
        }
    }
    // No normalization — matches training
}

void KeywordSpotter::runInference()
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> input_shape = { 1, 1, N_MELS, FRAMES };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        m_features.data(),
        m_features.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = { m_inputName.c_str() };
    const char* output_names[] = { m_outputName.c_str() };

    auto output_tensors = m_session.Run(
        Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* logits = output_tensors[0].GetTensorMutableData<float>();

    // Softmax for proper confidence
    float max_logit = *std::max_element(logits, logits + NUM_CLASSES);
    std::vector<float> probs(NUM_CLASSES);
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; ++i) {
        probs[i] /= sum;
    }

    int pred_idx = std::max_element(probs.begin(), probs.end()) - probs.begin();
    float confidence = probs[pred_idx];

    // Adjust threshold based on your model's performance (0.7–0.9 typical)
    if (confidence > 0.8f) {
        std::cout << "Detected: " << LABELS[pred_idx]
            << " (confidence: " << confidence << ")\n";
    }
}

int KeywordSpotter::argmax(const float* data, int size)
{
    return static_cast<int>(std::max_element(data, data + size) - data);
}

