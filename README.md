# SayIt
An AI-based voice recognition program that detects small words that we teach it.
Words consist of the following: ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
I use ONNX network to teach those words to the neural network. 
Then I use the trained model inside the C++ code. 
The trained network is kws.onnx  (KWS means KeyWord Spot)
