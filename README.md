# ForensicAI — Suspect Face Composite Generator

ForensicAI is a full-stack AI web application that generates photorealistic human faces from natural language descriptions to assist forensic investigations. It uses a combination of an NLP encoder (CLIP) and a generative adversarial network (StyleGAN2) to synthesize faces that match descriptive features.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.9 to 3.11 recommended.
- **Hardware**: A CUDA-compatible NVIDIA GPU is highly recommended for generating faces quickly. It will run on a CPU, but it will be slower.

### 2. Installation
Open a terminal in the project directory (`e:\rupeshmajor`) and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download the Pretrained AI Weights (Required for Photorealism)
The system requires the pretrained StyleGAN2-FFHQ weights (~363 MB) to generate realistic faces instead of procedural demos.
Run the included downloader script:

```bash
python download_weights.py
```
*(This script will download the weights and place them in the `e:\rupeshmajor\weights` directory).*

### 4. Run the Application
Start the Flask backend server:

```bash
python app.py
```

### 5. Access the Web Interface
1. Wait for the terminal to print `Starting Forensic AI server on port 5000...`.
2. Open your web browser (Chrome/Edge/Firefox).
3. Navigate to: **http://localhost:5000**
4. Enter a description and click "Generate Suspect Face"!

---

## 📂 Project Structure

- `app.py`: The main Flask backend server.
- `download_weights.py`: Utility script to download the StyleGAN2 FFHQ weights.
- `requirements.txt`: Python package dependencies.
- `models/`: Contains the AI neural network architectures (`stylegan2.py`, `text_encoder.py`, `latent_mapper.py`).
- `templates/`: Contains the `index.html` frontend interface.
- `static/`: Contains the CSS styling and JavaScript logic for the frontend.
- `logs/`: Contains the `generation.log` audit trail detailing every face generated.
- `weights/`: Directory where the ~363MB `stylegan2-ffhq-config-f.pt` file must reside.

---

## 🛠️ Troubleshooting

- **Server crashes immediately on startup**: Ensure you have installed all dependencies using `pip install -r requirements.txt`.
- **Generated faces are solid brown/blank blocks**: This means the pretrained weights are corrupted. Delete the `weights` folder and run `python download_weights.py` again.
- **Browser shows "Site cannot be reached"**: Ensure `app.py` is actively running in your terminal without any errors.
