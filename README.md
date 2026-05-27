# **[Qwen3-VL-abliterated-MAX-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-abliterated-MAX-Fast)**

Qwen3-VL-abliterated-MAX-Fast is an experimental, high-performance visual reasoning and optical character recognition (OCR) workspace. Powered by the unredacted `prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX` architecture, this suite is designed to deliver uncensored, raw image-to-text processing, precise text transcription, and intricate scene understanding. The web application features a standalone, custom-engineered interface built on vanilla web technologies via a headless Gradio framework. It provides a drag-and-drop media drop zone, inline state validation notifications, and a real-time raw output token stream. Optimized to harness active GPU acceleration with Flash Attention 2, it allows developers and security researchers to completely bypass structural system guardrails during complex visual reasoning evaluations.

 <img width="1920" height="1720" alt="Screenshot 2026-03-23 at 18-59-16 Qwen3-VL-abliterated-MAX-Fast - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/9136f9d1-47bb-44e7-aaec-5c0bf453d877" />

### **Key Features**

* **Abliterated Vision Engine:** Utilizes an unredacted version of the Qwen3-VL-4B-Instruct model to secure unrestricted vision-language processing and uncensored textual rendering from image inputs.
* **Custom Headless Interface:** Houses a sleek, dark terminal-inspired frontend layout designed with embedded JavaScript handling for real-time asset syncing and dynamic UI state response.
* **Streaming Token Output:** Displays responses step-by-step using text streamer loops that push chunks directly onto the output viewport as they are decoded by the transformer.
* **Advanced Pipeline Controls:** Offers modular option parameters to manually fine-tune token parameters including Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Streamlined Export Utilities:** Equipped with built-in instant click tools to quickly copy the entire output text block to the clipboard or download the raw layout response as a clean `.txt` file.

### **Repository Structure**

```text
├── images/
│   ├── 1.jpg
│   └── 2.jpg
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
└── requirements.txt

```

### **Installation and Requirements**

To configure Qwen3-VL-abliterated-MAX-Fast locally, set up an environment with the dependencies listed below. A system containing a modern, CUDA-compatible GPU is required for optimal inference speeds and Flash Attention execution.

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local package manager is upgraded to align with modern build conditions:

```bash
pip install pip>=26.1

```

**2. Install Core Dependencies**
Install the core deep learning stack, web layers, and document processing utilities:

```bash
pip install -r requirements.txt

```

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, which guarantees rapid virtual environment setup and deterministic dependency syncing.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast.git
cd Qwen3-VL-abliterated-MAX-Fast

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

### **Core Requirements List**

The application builds on the following core dependencies (defined in `requirements.txt`):

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.11.0
torchvision
matplotlib
pdf2image
requests
pymupdf
kernels
hf_xet
spaces
pillow
gradio==6.15.0
fpdf
timm
av

```

### **Usage**

Once the FastAPI web layout initiates, load the app locally by pointing your browser to the terminal endpoint (typically `http://127.0.0.1:7860/`).

1. **Upload Asset:** Drop an asset sheet, image document, page screenshot, or scene file into the dashed orange uploader area.
2. **Write Instruction:** Type a descriptive instruction directive inside the **Query Input** box (e.g., *"Read all visible text in the image"* or *"Describe the image in detail"*).
3. **Advanced Settings (Optional):** Tweak advanced sampler configurations like Temperature, Top-p, and Repetition Penalty to adjust output variations.
4. **Execute:** Click **Run Inference** to pass parameters to the backend. The results will immediately start streaming into the **Raw Output Stream** viewport.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast.git](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast.git)
