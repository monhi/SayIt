#include "CWaveIn.h"
#include <iostream>
#include <algorithm>

CWaveIn::CWaveIn(size_t bufferSeconds)
    : SAMPLE_RATE(16000),
    NUM_CHANNELS(1),
    BITS_PER_SAMPLE(16),
    CHUNK_MS(64),
    running(false),
    circularBuffer(bufferSeconds* SAMPLE_RATE),
    bufferFront(0),
    bufferEnd(0)
{}

CWaveIn::~CWaveIn() {
    stop();
}

bool CWaveIn::start() {
    std::lock_guard<std::mutex> lock(mutex);
    if (running) return false;

    WAVEFORMATEX wfx{};
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = NUM_CHANNELS;
    wfx.nSamplesPerSec = SAMPLE_RATE;
    wfx.wBitsPerSample = BITS_PER_SAMPLE;
    wfx.nBlockAlign = (wfx.wBitsPerSample / 8) * wfx.nChannels;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
    wfx.cbSize = 0;

    if (waveInOpen(&hWaveIn, WAVE_MAPPER, &wfx,
        (DWORD_PTR)waveInCallback, (DWORD_PTR)this,
        CALLBACK_FUNCTION) != MMSYSERR_NOERROR)
    {
        std::cerr << "Failed to open microphone." << std::endl;
        return false;
    }

    chunkSamples = SAMPLE_RATE * CHUNK_MS / 1000;
    tempBuffer.resize(chunkSamples);

    ZeroMemory(&waveHeader, sizeof(WAVEHDR));
    waveHeader.lpData = reinterpret_cast<LPSTR>(tempBuffer.data());
    waveHeader.dwBufferLength = static_cast<DWORD>(tempBuffer.size() * sizeof(short));
    waveHeader.dwFlags = 0;
    waveHeader.dwLoops = 0;

    if (waveInPrepareHeader(hWaveIn, &waveHeader, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
        std::cerr << "Failed to prepare header." << std::endl;
        waveInClose(hWaveIn);
        return false;
    }

    if (waveInAddBuffer(hWaveIn, &waveHeader, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
        std::cerr << "Failed to add buffer." << std::endl;
        waveInClose(hWaveIn);
        return false;
    }

    if (waveInStart(hWaveIn) != MMSYSERR_NOERROR) {
        std::cerr << "Failed to start recording." << std::endl;
        waveInClose(hWaveIn);
        return false;
    }

    running = true;
    captureThread = std::thread([this]() { this->processingThread(); });

    return true;
}

void CWaveIn::stop() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!running) return;

    running = false;
    if (captureThread.joinable())
        captureThread.join();

    waveInStop(hWaveIn);
    waveInUnprepareHeader(hWaveIn, &waveHeader, sizeof(WAVEHDR));
    waveInClose(hWaveIn);
}

size_t CWaveIn::getAudioData(std::vector<float>& outBuffer, size_t maxSamples) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    size_t available = (bufferEnd + circularBuffer.size() - bufferFront) % circularBuffer.size();
    size_t toCopy = min(maxSamples, available);
    outBuffer.resize(toCopy);

    for (size_t i = 0; i < toCopy; ++i) {
        outBuffer[i] = circularBuffer[(bufferFront + i) % circularBuffer.size()];
    }
    bufferFront = (bufferFront + toCopy) % circularBuffer.size();
    return toCopy;
}

size_t CWaveIn::getBufferUsage() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    return (bufferEnd + circularBuffer.size() - bufferFront) % circularBuffer.size();
}

void CALLBACK CWaveIn::waveInCallback(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance,
    DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    if (uMsg == WIM_DATA) {
        CWaveIn* pThis = reinterpret_cast<CWaveIn*>(dwInstance);
        pThis->handleData(reinterpret_cast<WAVEHDR*>(dwParam1));
    }
}

void CWaveIn::handleData(WAVEHDR* header) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    short* data = reinterpret_cast<short*>(header->lpData);

    for (DWORD i = 0; i < header->dwBytesRecorded / sizeof(short); ++i) {
        // Convert short PCM to float [-1,1]
        float sample = static_cast<float>(data[i]) / 32768.0f;
        circularBuffer[bufferEnd] = sample;
        bufferEnd = (bufferEnd + 1) % circularBuffer.size();

        // Overwrite old data if buffer full
        if (bufferEnd == bufferFront) {
            bufferFront = (bufferFront + 1) % circularBuffer.size();
        }
    }

    // Re-add buffer for next capture
    waveInAddBuffer(hWaveIn, header, sizeof(WAVEHDR));
}

void CWaveIn::processingThread() {
    while (running) {
        Sleep(10);
    }
}
