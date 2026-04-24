"""
Facial Composite Generation System — Flask Backend
Generates realistic face composites from natural language descriptions.
"""

import os
import sys
import json
import time
import base64
import logging
from io import BytesIO
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.text_encoder import TextEncoder
from models.latent_mapper import LatentMapper
from models.stylegan2 import StyleGAN2Generator
from utils.preprocessing import build_description, validate_input

import torch

# ─── App Configuration ───────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ─── Structured Logging ─────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Console logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('forensic_ai')

# File logger for audit trail
audit_handler = logging.FileHandler(os.path.join(LOG_DIR, 'generation.log'), encoding='utf-8')
audit_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
audit_logger = logging.getLogger('audit')
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


# ─── Model Initialization ───────────────────────────────────────────
logger.info("Initializing AI models...")
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
logger.info(f"Using device: {device}")

text_encoder = TextEncoder(device=device)
latent_mapper = LatentMapper()
latent_mapper.eval()
generator = StyleGAN2Generator(device=device)

logger.info("All models initialized.")


# ─── Helper Functions ────────────────────────────────────────────────
def image_to_base64(img) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='PNG', quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def log_generation(attributes, description, num_variations, generation_time, scores):
    """Log generation request for audit trail."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'attributes': attributes,
        'description': description,
        'num_variations': num_variations,
        'generation_time_ms': round(generation_time * 1000, 2),
        'similarity_scores': scores,
    }
    audit_logger.info(json.dumps(entry, ensure_ascii=False))


# ─── API Endpoints ───────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'device': device,
        'models': {
            'text_encoder': 'CLIP' if text_encoder.is_loaded else 'demo',
            'latent_mapper': 'initialized',
            'generator': 'StyleGAN2' if generator.is_loaded else 'demo',
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/generate-face', methods=['POST'])
def generate_face():
    """
    Generate face composites from description.

    Request JSON:
        gender, age_group, hair_style, facial_hair, accessories,
        skin_tone, face_shape, eye_color, description

    Response JSON:
        images: list of base64 image strings
        attributes: dict of input attributes
        description: composed NL description
        similarity_scores: list of CLIP similarity scores
        generation_time_ms: float
    """
    try:
        data = request.get_json(force=True)

        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        start_time = time.time()

        # Extract attributes
        attributes = {
            'gender': data.get('gender', ''),
            'age_group': data.get('age_group', ''),
            'hair_style': data.get('hair_style', ''),
            'facial_hair': data.get('facial_hair', ''),
            'accessories': data.get('accessories', ''),
            'skin_tone': data.get('skin_tone', ''),
            'face_shape': data.get('face_shape', ''),
            'eye_color': data.get('eye_color', ''),
            'description': data.get('description', ''),
        }

        # Build natural language description
        nl_description = build_description(attributes)
        logger.info(f"Processing: {nl_description}")

        # Step 1: Encode text
        text_embedding = text_encoder.encode(nl_description)

        # Step 2: Map to initial latent space
        with torch.no_grad():
            latent_vector = latent_mapper(text_embedding)

        # Step 3: Optimize latent vector for semantic alignment (Zero-Shot)
        if hasattr(generator, 'optimize_latent') and generator.is_loaded:
            # Re-enable optimization to apply text constraints
            # Use fewer steps and higher learning rate on MPS/CPU to prevent hanging
            optim_steps = 40 if device == 'cuda' else 5
            lr_rate = 0.08 if device == 'cuda' else 0.20
            
            latent_vector = generator.optimize_latent(
                text_embedding=text_embedding,
                text_encoder=text_encoder,
                steps=optim_steps,
                lr=lr_rate
            )

        # Step 4: Generate multiple face variations 
        num_variations = 3
        face_images = generator.generate_variations(latent_vector, num_variations)

        # Step 4: Compute similarity scores
        similarity_scores = []
        base64_images = []
        for img in face_images:
            score = text_encoder.compute_similarity(text_embedding, img)
            similarity_scores.append(score)
            base64_images.append(image_to_base64(img))

        generation_time = time.time() - start_time

        # Log for audit trail
        log_generation(attributes, nl_description, num_variations, generation_time, similarity_scores)

        logger.info(f"Generated {num_variations} variations in {generation_time:.2f}s | Scores: {similarity_scores}")

        return jsonify({
            'images': base64_images,
            'attributes': attributes,
            'description': nl_description,
            'similarity_scores': similarity_scores,
            'generation_time_ms': round(generation_time * 1000, 2),
        })

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Forensic AI server...")
    logger.info(f"Server is running! Click this link to open: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
