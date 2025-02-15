

# Sliding Tile Atteniton Kernel

<div align="center">
<img src=../../assets/sliding_tile_attn_map.png width="80%"/>
</div>

## Installation
We test our code on Pytorch 2.5.1 and CUDA 12.4. Currently we only have implementation on H100.
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
python setup.py install
```

## Usage

```python
from st_attn import sliding_tile_attention
# assuming video size (T, H, W) = (30, 48, 80), text tokens = 256 with padding. 
# q, k, v: [batch_size, num_heads, seq_length, head_dim], seq_length = T*H*W + 256
# a tile is a cube of size (6, 8, 8)
# window_size in tiles: [(window_t, window_h, window_w), (..)...]. For example, window size (3, 3, 3) means a query can attend to (3x6, 3x8, 3x8) = (18, 24, 24) tokens out of the total 30x48x80 video.
# text_length: int ranging from 0 to 256
out = sliding_tile_attention(q, k, v, window_size, text_length)
```


## Test
```bash
python test/test_sta.py
```

## Use STA with FastVideo


## Acknowledgement

We learned or reuse code from FlexAtteniton, NATEN, and ThunderKittens.