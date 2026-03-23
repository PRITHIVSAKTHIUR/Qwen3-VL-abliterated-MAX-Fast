# **Qwen3-VL-abliterated-MAX-Fast**

Qwen3-VL-abliterated-MAX-Fast is an experimental, unredacted visual reasoning and optical character recognition suite. Powered by the specialized `prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX` model, this application provides an uncensored, highly capable environment for profound image analysis, detailed scene description, and raw text extraction. The suite is wrapped in a bespoke, responsive web interface built with custom HTML, CSS, and JavaScript, facilitating a seamless drag-and-drop workflow for media uploads. Fully optimized for CUDA-enabled GPUs utilizing Flash Attention 3, Qwen3-VL-abliterated-MAX-Fast grants developers and researchers unrestricted control over generation parameters, making it an ideal sandbox for testing raw, unfiltered multimodal AI capabilities.

### **Key Features**

* **Unredacted Model Architecture:** Utilizes a specifically tuned, unredacted version of Qwen3-VL-4B-Instruct, designed for raw visual understanding and uncensored image-to-text generation.
* **Custom User Interface:** Features a bespoke, highly responsive Gradio frontend. It includes a sleek drag-and-drop media zone, real-time output text streaming, and an integrated advanced settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by manually adjusting text generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in utility actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a local `.txt` file.
* **Flash Attention 3 Integration:** Employs `kernels-community/flash-attn3` for maximized, memory-efficient inference speeds on compatible modern GPU hardware.

### **Repository Structure**

```text
├── images/
│   ├── 1.jpg
│   └── 2.jpg
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run Qwen3-VL-abliterated-MAX-Fast locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
pdf2image
requests
pymupdf
kernels
hf_xet
spaces
pillow
gradio
fpdf
timm
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the model. Note that the model will be downloaded and loaded into VRAM upon its first invocation.

### **License and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast.git](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-abliterated-MAX-Fast.git)
