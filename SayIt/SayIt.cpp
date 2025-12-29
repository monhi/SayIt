#include "CWaveIn.h"
#include <conio.h>
#include <iostream>
#include <vector>
#include <string>
#include "CKeywordSpotter.h"

int main() 
{
    TCHAR lpFileName[MAX_PATH] = {0};
    GetModuleFileName(nullptr, lpFileName, MAX_PATH);
    std::wstring stemp = lpFileName;
    stemp  = stemp.substr(0, stemp.find_last_of('\\')+1);
    stemp += L"kws.onnx";
    
    CKeywordSpotter kws(stemp.c_str());
    CWaveIn mic(10);
    if (!mic.start()) return 1;

    std::cout << "Recording... Press ENTER to stop." << std::endl;

    while (true) {
        if (_kbhit() && _getch() == 13) break;

        std::vector<float> audioChunk;
        size_t n = mic.getAudioData(audioChunk, 1024);
        if (n > 0) 
        {
            std::cout << "." ;
             //std::cout << "Samples read: " << n << std::endl;
            // Send audioChunk directly to ONNX feature extraction
            kws.processAudio(audioChunk);
        }
        Sleep(2);
    }
    mic.stop();
    return 0;
}
