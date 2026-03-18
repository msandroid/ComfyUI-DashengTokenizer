# ComfyUI-DashengTokenizer

ComfyUI custom nodes for [mispeech/dashengtokenizer](https://huggingface.co/mispeech/dashengtokenizer) on Hugging Face. DashengTokenizer is a continuous audio tokenizer for audio understanding and generation (encode audio to embeddings, decode embeddings back to audio). Optimized for **16 kHz mono** input and output.

## Installation

1. **Clone into ComfyUI custom_nodes**:
   ```bash
   cd path/to/ComfyUI/custom_nodes
   git clone https://github.com/msandroid/ComfyUI-DashengTokenizer.git
   ```

2. **Install dependencies** in the ComfyUI Python environment:
   ```bash
   pip install -r ComfyUI-DashengTokenizer/comfyui_dashengtokenizer/requirements.txt
   ```
   If ComfyUI is run via Stability Matrix, use that environment's `pip` (e.g. `Data\Packages\ComfyUI\venv\Scripts\pip.exe`).

3. Restart ComfyUI. The nodes appear under the **audio/dashengtokenizer** category.

## Nodes

| Node | Description |
|------|-------------|
| **Load DashengTokenizer Model** | Loads the model from Hugging Face (default: `mispeech/dashengtokenizer`). Optional input: `model_id` (string). Output: `model` (DASHENG_MODEL) to connect to Encode/Decode. |
| **DashengTokenizer Encode** | Input: `model` (DASHENG_MODEL), `audio` (AUDIO). Converts input audio to 16 kHz mono, then encodes to embeddings. Output: `embeddings` (DASHENG_EMBEDDINGS). |
| **DashengTokenizer Decode** | Input: `model` (DASHENG_MODEL), `embeddings` (DASHENG_EMBEDDINGS). Decodes embeddings to audio. Output: `audio` (AUDIO) at 16 kHz mono. |

Typical workflow: **Load Model** -> **Encode** (with an AUDIO source) -> **Decode** to get reconstructed AUDIO. You can use the embeddings for other pipelines (e.g. conditioning, analysis) before decoding.

## Limitations

- The model is optimized for **16 kHz mono** audio. Other sample rates and stereo are resampled/converted automatically at the Encode node.
- First run will download the model from Hugging Face (cached afterward).

## Model and license

- **Model**: [mispeech/dashengtokenizer](https://huggingface.co/mispeech/dashengtokenizer) (Horizon Team, Xiaomi MiLM Plus).
- **License**: Apache 2.0.
