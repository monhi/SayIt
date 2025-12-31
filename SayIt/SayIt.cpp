#include "CWaveIn.h"
#include <conio.h>
#include <iostream>
#include <vector>
#include <string>
#include "CKeywordSpotter.h"
#include <fstream>
#include "WavReader.h"

int main(int argc, char* argv[])
{
    TCHAR lpFileName[MAX_PATH] = {0};
    GetModuleFileName(nullptr, lpFileName, MAX_PATH);
    std::wstring stemp = lpFileName;
    stemp  = stemp.substr(0, stemp.find_last_of('\\')+1);
    stemp += L"kws.onnx";
    KeywordSpotter kws(stemp.c_str());

    std::cout << "This program should detect these words: down,go,left,no,right,stop,up,yes" << std::endl;

    if ( argc > 1 )
    {
        std::vector<float> wav;
        if (!WavReader::loadMono16k(argv[1], wav))
        {
            return 1;
        }            

        std::cout << "Offline WAV test: " << argv[1] << std::endl;
        kws.processWavFile(wav);
        return 0;
    }

    std::string filename = "monhi123.raw";
    std::ofstream file(filename, std::ios::binary);

    CWaveIn mic(10);
    if (!mic.start()) return 1;
      
    std::cout << "Recording... Press ENTER to stop." << std::endl;

    while (true) 
    {
        if (_kbhit() && _getch() == 13) break;

        std::vector<float> audioChunk;
        size_t n = mic.getAudioData(audioChunk, 4096);
        if (n > 0) 
        {
            // std::cout << n << ",";
            // save data into a file.
            // std::cout << "Samples read: " << n << std::endl;
            // Send audioChunk directly to ONNX feature extraction
            file.write(reinterpret_cast<const char*>(audioChunk.data()),audioChunk.size() * sizeof(float));
            kws.processAudio(audioChunk);
        }
        Sleep(2);
    }
    mic.stop();
    return 0;
}
