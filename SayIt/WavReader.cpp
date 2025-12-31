#include "WavReader.h"
#include <fstream>
#include <iostream>

#pragma pack(push,1)
struct WavHeader
{
    char riff[4];
    int fileSize;
    char wave[4];
    char fmt[4];
    int fmtSize;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    char data[4];
    int dataSize;
};
#pragma pack(pop)

bool WavReader::loadMono16k(const std::string& path, std::vector<float>& samples)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::cerr << "Cannot open WAV file\n";
        return false;
    }

    WavHeader h{};
    f.read((char*)&h, sizeof(h));

    if (strncmp(h.riff, "RIFF", 4) != 0 ||
        strncmp(h.wave, "WAVE", 4) != 0)
    {
        std::cerr << "Invalid WAV format\n";
        return false;
    }

    if (h.sampleRate != 16000 || h.numChannels != 1 || h.bitsPerSample != 16)
    {
        std::cerr << "WAV must be 16kHz, mono, 16-bit PCM\n";
        return false;
    }

    int numSamples = h.dataSize / 2;
    samples.resize(numSamples);

    for (int i = 0; i < numSamples; i++)
    {
        short s;
        f.read((char*)&s, sizeof(short));
        samples[i] = s / 32768.0f;
    }

    return true;
}
