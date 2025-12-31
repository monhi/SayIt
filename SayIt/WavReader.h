#pragma once
#include <vector>
#include <string>

class WavReader
{
public:
    static bool loadMono16k(const std::string& path, std::vector<float>& samples);
};
