BiConvGRU: Bidirectional Convolutional Gated Recurrent Unit


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

git clone [BiConvGRU](https://github.com/TopDianyou/BiConvGRU)
cd BiConvGRU
pip install -r requirements.txt

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


