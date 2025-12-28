# SayIt
An AI-based voice recognition program that detects small words that we teach it.<br/>
Words consist of the following: ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'].<br/>
I use ONNX network to teach those words to the neural network.<br/>
Then I use the trained model inside the C++ code.<br/>
The trained network is kws.onnx  (KWS means KeyWord Spot)<br/>
