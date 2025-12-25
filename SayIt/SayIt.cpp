#include "CWaveIn.h"
#include <conio.h>
#include <iostream>
#include <vector>

int main() {
    CWaveIn mic(10);
    if (!mic.start()) return 1;

    std::cout << "Recording... Press ENTER to stop." << std::endl;

    while (true) {
        if (_kbhit() && _getch() == 13) break;

        std::vector<float> audioChunk;
        size_t n = mic.getAudioData(audioChunk, 1024);
        if (n > 0) 
        {
            std::cout << "Samples read: " << n << std::endl;
            // Send audioChunk directly to ONNX feature extraction
        }

        Sleep(10);
    }

    mic.stop();
    return 0;
}
