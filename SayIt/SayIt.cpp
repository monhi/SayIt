#include <iostream>
#include <vector>
#include <conio.h> // For _kbhit() and _getch()
#include "CWaveIn.h"

int main() 
{
    CWaveIn mic(10); // 10-second circular buffer
    if (!mic.start()) {
        std::cerr << "Failed to start microphone!" << std::endl;
        return 1;
    }

    std::cout << "Recording... Press ENTER to stop." << std::endl;

    bool running = true;
    while (running) {
        // Check if user pressed a key
        if (_kbhit()) {
            int ch = _getch();
            if (ch == 13) { // Enter key
                running = false;
                break;
            }
        }

        // Process audio data
        std::vector<short> audioChunk;
        size_t samplesRead = mic.getAudioData(audioChunk, 1024); // read up to 1024 samples
        if (samplesRead > 0) 
        {
            // Example: just print number of samples retrieved
            // std::cout << "Samples read from buffer: " << samplesRead << std::endl;
            // TODO: insert your audio processing / keyword recognition here
        }
        // Small sleep to avoid CPU hog
        Sleep(10);
    }
    mic.stop();
    std::cout << "Recording stopped." << std::endl;
    return 0;
}
