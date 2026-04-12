# Weight Format

The Python `export.py` script produces two artefacts:

```
<output_dir>/
├── manifest.json
└── weights/
    ├── encoder_0_conv1_conv_weight.bin
    ├── encoder_0_conv1_bn_weight.bin
    ├── ...
    └── head_bias.bin
```

---

## manifest.json

Top-level structure:

```json
{
  "version": "1.0",
  "dtype":   "float16",
  "architecture": { ... },
  "layers": [ ... ]
}
```

### `architecture` object

| Field | Type | Description |
|---|---|---|
| `type` | string | `"nafnet_residual"` or `"nafnet_temporal"` |
| `enc_channels` | int[] | Channel widths per encoder level, e.g. `[64,128,256,512]` |
| `num_levels` | int | Length of `enc_channels` |
| `in_channels` | int | Input image channels (3 for RGB) |
| `out_channels` | int | Output channels (3 for RGB) |
| `num_frames` | int | Temporal window (NAFNetTemporal only) |
| `use_warp` | bool | Optional learned warp path (NAFNetTemporal only) |

### `layers` array

Each entry describes one tensor:

```json
{
  "name":  "encoders.0.conv1.conv.weight",
  "shape": [64, 3, 3, 3],
  "dtype": "float16",
  "file":  "weights/encoders_0_conv1_conv_weight.bin"
}
```

| Field | Description |
|---|---|
| `name` | PyTorch `state_dict` key — must match the C++ layer constructor's expected key |
| `shape` | Tensor dimensions, NCHW for 4-D weight tensors |
| `dtype` | `"float16"` or `"float32"` |
| `file` | Path to the `.bin` file, relative to the manifest |

---

## .bin files

Raw, headerless binary arrays:

- **Byte order**: little-endian (x86/ARM native; no conversion needed on any modern platform)
- **Layout**: C-contiguous (row-major), matching the `shape` field in the manifest
- **Dtype**:
  - Conv / linear weights and biases: `float16` (2 bytes per element)
  - BN `weight`, `bias`, `running_mean`, `running_var`: always `float32` (4 bytes per element), regardless of the model's `dtype` field

### Reading a `.bin` file in Python

```python
import numpy as np

arr = np.fromfile("weights/encoders_0_conv1_conv_weight.bin", dtype=np.float16)
arr = arr.reshape([64, 3, 3, 3])   # shape from manifest
```

### Reading a `.bin` file in C++

The `WeightStore` handles this automatically via `mmap` + lazy H2D upload.  For manual inspection:

```cpp
// Read raw bytes and reinterpret as __half
std::ifstream f("weights/encoders_0_conv1_conv_weight.bin", std::ios::binary);
std::vector<__half> buf(64 * 3 * 3 * 3);
f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(__half));
```

---

## Weight name conventions

The C++ engine constructs weight names from the exported NAFNet pattern.  The table below shows the mapping for the spatial model:

| PyTorch path | C++ lookup key |
|---|---|
| `encoders.{lvl}.conv1.conv.weight` | `encoders.{lvl}.conv1.conv.weight` |
| `encoders.{lvl}.conv1.bn.weight` | `encoders.{lvl}.conv1.bn.weight` |
| `encoders.{lvl}.conv1.bn.bias` | `encoders.{lvl}.conv1.bn.bias` |
| `encoders.{lvl}.conv1.bn.running_mean` | `encoders.{lvl}.conv1.bn.running_mean` |
| `encoders.{lvl}.conv1.bn.running_var` | `encoders.{lvl}.conv1.bn.running_var` |
| `bottleneck.{0,1}.conv.weight` | `bottleneck.{0,1}.conv.weight` |
| `bottleneck.{0,1}.bn.*` | `bottleneck.{0,1}.bn.*` |
| `decoders.{lvl}.conv{1,2}.conv.weight` | `decoders.{lvl}.conv{1,2}.conv.weight` |
| `head.weight` | `head.weight` |
| `head.bias` | `head.bias` |

Additional keys for NAFNetTemporal:

| PyTorch path | C++ lookup key |
|---|---|
| `align_layers.{lvl}.offset_conv.weight` | `align_layers.{lvl}.offset_conv.weight` |
| `align_layers.{lvl}.offset_conv.bias` | `align_layers.{lvl}.offset_conv.bias` |
| `align_layers.{lvl}.mask_conv.weight` | `align_layers.{lvl}.mask_conv.weight` |
| `align_layers.{lvl}.mask_conv.bias` | `align_layers.{lvl}.mask_conv.bias` |
| `align_layers.{lvl}.weight` | `align_layers.{lvl}.weight` |
| `align_layers.{lvl}.bias` | `align_layers.{lvl}.bias` |
| `fusion_layers.{lvl}.weight` | `fusion_layers.{lvl}.weight` |
| `fusion_layers.{lvl}.bias` | `fusion_layers.{lvl}.bias` |

---

## Version history

| Version | Change |
|---|---|
| `1.0` | Initial format — FP16 conv weights, FP32 BN stats, NCHW layout |
