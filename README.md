BiConvGRU: Bidirectional Convolutional Gated Recurrent Unit
PyTorch
License: MIT

BiConvGRU is a Bidirectional Convolutional GRU implemented in PyTorch, designed for spatiotemporal sequence modeling (e.g., video analysis, weather forecasting, medical imaging). It combines convolutional spatial feature extraction with GRU-based temporal modeling, supporting multi-layer cascading and bidirectional processing for context-aware predictions.

Features
Bidirectional Spatiotemporal Modeling
Captures forward and backward temporal dependencies for enhanced context awareness.

Flexible Architecture
Customizable input/hidden dimensions, kernel sizes, layer depth, downsampling, and dropout.

Multi-Layer Support
Deep feature extraction via num_layers parameter.

Dynamic Sequence Handling
Processes variable-length input sequences efficiently.

GPU Acceleration
Optimized with PyTorch primitives for seamless CUDA support.

Installation
Dependencies:

Python 3.6+

PyTorch 1.9+

bash
git clone https://github.com/yourusername/BiConvGRU.git
cd BiConvGRU
pip install -r requirements.txt
Quick Start
Initialize Model
python
import torch
from bigru import BiConvGRU

model = BiConvGRU(
    input_size=(64, 64),    # Input spatial dimensions (H, W)
    input_dim=3,            # Input channels (e.g., RGB)
    hidden_dim=[32, 64],    # Hidden channels per layer
    kernel_size=[3, 5],     # Convolutional kernel sizes per layer
    num_layers=2,           # Number of BiConvGRU layers
    bidirectional=True,     # Enable bidirectional processing
    dropout=0.2,            # Inter-layer dropout
    output_size=(32, 32)    # Downsampled output dimensions
).to("cuda")
Forward Pass
python
# Input shape: (batch_size, seq_len, channels, height, width)
inputs = torch.randn(4, 8, 3, 64, 64).to("cuda")

# Returns sequence outputs and final states
outputs, states = model(inputs, return_sequence=True)

print("Output shape:", outputs[0].shape)       # [4, 8, 128, 32, 32] (channels doubled for bidirectional)
print("Final hidden state:", states[0][-1].shape)  # [4, 64, 32, 32]
Parameters
BiConvGRU Arguments
Parameter	Type	Description
input_size	Tuple[int, int]	Spatial dimensions of input (H, W)
input_dim	int	Number of input channels
hidden_dim	Union[int, List[int]]	Hidden channels per layer (int or list)
kernel_size	Union[int, Tuple, List]	Convolutional kernel size (int/tuple/list)
num_layers	int	Number of stacked BiConvGRU layers
bidirectional	bool	Enable bidirectional processing
batch_first	bool	If True, input shape is (batch, seq, *)
dropout	float	Dropout probability between layers
output_size	Tuple[int, int]	Downsampled output dimensions (optional)
Use Cases
Video Action Recognition
Model spatiotemporal dependencies in video frames.

Weather Prediction
Forecast future states from radar echo sequences.

Medical Imaging
Track dynamic changes in ultrasound/CT scans.

Autonomous Driving
Predict trajectories of surrounding objects.

Architecture
Simplified Diagram
Input (Seq, C, H, W)
  │
  ▼
[Bidirectional ConvGRU Layers]  
  │
  ▼
Downsample (Optional)
  │
  ▼
Output (Seq, Hidden*2, H', W')  # Hidden*2 for bidirectional
Key Components
ConvGRUCell
Single-layer ConvGRU with gated convolutions and state updates.

ConvGRU
Multi-layer unidirectional ConvGRU with adaptive downsampling.

BiConvGRU
Bidirectional wrapper with forward/backward processing.

Contributing
Contributions are welcome!

Submit issues for bug reports or feature requests.

Open PRs for improvements (follow PEP8 and include tests).

Citation
If this work aids your research, please cite:

bibtex
@software{BiConvGRU,
  author = {Dianyou Yu},
  title = {BiConvGRU: Bidirectional Convolutional GRU for Spatiotemporal Modeling},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/BiConvGRU}}
}
License
MIT License

Add performance benchmarks or visualization examples if available. Adjust the content to match your implementation details.
