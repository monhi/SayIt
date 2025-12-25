#pragma once
#include <windows.h>
#include <mmsystem.h>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#pragma comment(lib, "winmm.lib")

class CWaveIn {
public:
    CWaveIn(size_t bufferSeconds = 10);
    ~CWaveIn();

    // Start capturing
    bool start();

    // Stop capturing
    void stop();

    // Get audio data from circular buffer as float [-1,1]
    // Returns number of samples copied
    size_t getAudioData(std::vector<float>& outBuffer, size_t maxSamples);

    // Get current buffer usage
    size_t getBufferUsage();

private:
    const int SAMPLE_RATE;
    const int NUM_CHANNELS;
    const int BITS_PER_SAMPLE;
    const int CHUNK_MS;

    HWAVEIN hWaveIn = nullptr;
    WAVEHDR waveHeader{};
    std::vector<short> tempBuffer;
    size_t chunkSamples;

    std::vector<float> circularBuffer;
    size_t bufferFront;
    size_t bufferEnd;
    std::mutex bufferMutex;

    std::thread captureThread;
    std::mutex mutex;
    std::atomic<bool> running;

    // WaveIn callback
    static void CALLBACK waveInCallback(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance,
        DWORD_PTR dwParam1, DWORD_PTR dwParam2);

    void handleData(WAVEHDR* header);
    void processingThread();
};
