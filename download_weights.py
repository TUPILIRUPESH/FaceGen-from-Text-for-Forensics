"""
StyleGAN2 Pretrained Weights Downloader
Downloads FFHQ-trained StyleGAN2 weights for realistic face generation.
"""

import os
import sys
import hashlib

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'stylegan2-ffhq-config-f.pt')

# Google Drive file ID for rosinality's converted StyleGAN2-FFHQ weights
# This is the config-f model trained on FFHQ at 1024x1024
GDRIVE_ID = '1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_ID}'

# Alternative: Direct link (if gdown fails)
ALTERNATIVE_URLS = [
    'https://huggingface.co/aharley/stylegan2/resolve/main/stylegan2-ffhq-config-f.pt',
]


def download_weights(force=False):
    """
    Download StyleGAN2-FFHQ pretrained weights.

    Args:
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded weights file, or None on failure.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(WEIGHTS_FILE) and not force:
        size_mb = os.path.getsize(WEIGHTS_FILE) / (1024 * 1024)
        if size_mb > 100:  # Sanity check — file should be > 100MB
            print(f"[Download] Weights already exist at {WEIGHTS_FILE} ({size_mb:.1f} MB)")
            return WEIGHTS_FILE
        else:
            print(f"[Download] Existing file too small ({size_mb:.1f} MB), re-downloading...")

    # Try gdown first
    if _try_gdown():
        return WEIGHTS_FILE

    # Try alternative URLs
    for url in ALTERNATIVE_URLS:
        if _try_urllib(url):
            return WEIGHTS_FILE

    print("\n[Download] ⚠ Automatic download failed.")
    print(f"[Download] Please manually download the weights and place them at:")
    print(f"[Download]   {WEIGHTS_FILE}")
    print(f"[Download]")
    print(f"[Download] Download from one of these sources:")
    print(f"[Download]   1. https://drive.google.com/uc?id={GDRIVE_ID}")
    for url in ALTERNATIVE_URLS:
        print(f"[Download]   2. {url}")

    return None


def _try_gdown():
    """Try downloading with gdown (Google Drive downloader)."""
    try:
        import gdown
        print(f"[Download] Downloading StyleGAN2-FFHQ weights from Google Drive...")
        print(f"[Download] This may take a few minutes (~370 MB)...")
        gdown.download(GDRIVE_URL, WEIGHTS_FILE, quiet=False)

        if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 100 * 1024 * 1024:
            print(f"[Download] ✓ Successfully downloaded to {WEIGHTS_FILE}")
            return True
        else:
            print("[Download] gdown download appears incomplete")
            return False
    except ImportError:
        print("[Download] gdown not installed. Install with: pip install gdown")
        return False
    except Exception as e:
        print(f"[Download] gdown failed: {e}")
        return False


def _try_urllib(url):
    """Try downloading with urllib."""
    try:
        import urllib.request
        print(f"[Download] Trying: {url}")
        print(f"[Download] This may take a few minutes (~370 MB)...")

        def _progress(count, block_size, total_size):
            percent = min(100, count * block_size * 100 // total_size)
            mb = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r[Download] {percent}% ({mb:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, WEIGHTS_FILE, reporthook=_progress)
        print()

        if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 100 * 1024 * 1024:
            print(f"[Download] ✓ Successfully downloaded to {WEIGHTS_FILE}")
            return True
        else:
            print("[Download] Download appears incomplete")
            if os.path.exists(WEIGHTS_FILE):
                os.remove(WEIGHTS_FILE)
            return False
    except Exception as e:
        print(f"[Download] urllib failed: {e}")
        if os.path.exists(WEIGHTS_FILE):
            os.remove(WEIGHTS_FILE)
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download StyleGAN2 pretrained weights')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    args = parser.parse_args()

    result = download_weights(force=args.force)
    if result:
        print(f"\n✓ Weights ready at: {result}")
        print(f"  Size: {os.path.getsize(result) / (1024*1024):.1f} MB")
    else:
        print("\n✗ Download failed. Please download manually.")
        sys.exit(1)
