"""
Model handler for deployment - handles model loading with various sources
"""
import streamlit as st
import requests
import os
from pathlib import Path
import tempfile
import hashlib

@st.cache_resource
def download_model_from_url(model_url: str, expected_hash: str = None):
    """Download model file from external URL with caching"""
    try:
        # Create a cache directory
        cache_dir = Path(tempfile.gettempdir()) / "model_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Generate filename based on URL hash
        url_hash = hashlib.md5(model_url.encode()).hexdigest()[:8]
        model_path = cache_dir / f"model_{url_hash}.pth"
        
        if not model_path.exists():
            with st.spinner("⬇️ Downloading model file... This may take a few minutes."):
                st.info(f"📥 Downloading from: {model_url}")
                
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                st.progress(progress)
                
                # Verify file size
                file_size = model_path.stat().st_size
                if file_size < 1024 * 1024:  # Less than 1MB seems suspicious
                    model_path.unlink()  # Delete suspicious file
                    raise ValueError(f"Downloaded file too small: {file_size} bytes")
                
                st.success(f"✅ Model downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
        
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return None

def get_model_path():
    """Get model path with deployment handling"""
    
    # Option 1: Check for environment variable (for Streamlit Cloud)
    model_url = os.getenv("MODEL_URL")
    if model_url:
        st.info("🌐 Using model from environment URL")
        return download_model_from_url(model_url)
    
    # Option 2: Check for local model files (development)
    possible_models = [
        "best_staged_model.pth",
        "best_staged_model (1).pth",
        "model.pth",
        "checkpoint.pth"
    ]
    
    for model in possible_models:
        if Path(model).exists():
            st.success(f"✅ Using local model: {model}")
            return model
    
    # Option 3: Demo mode without model
    st.warning("""
    🚨 **Model Configuration Required**
    
    For full functionality, you need a trained model file. Options:
    
    ### For Deployment (Streamlit Cloud):
    1. **Upload your model** to Google Drive, Dropbox, or GitHub Releases
    2. **Get a direct download URL** (for Google Drive, use: `https://drive.google.com/uc?id=FILE_ID`)
    3. **Add environment variable** in Streamlit Cloud: `MODEL_URL=your_download_url`
    
    ### For Local Development:
    - Place `best_staged_model.pth` in the project root directory
    
    ### Demo Mode:
    - The app will run in demo mode with limited functionality
    """)
    
    return None

def show_model_info(model_path):
    """Display model information"""
    if model_path and Path(model_path).exists():
        file_size = Path(model_path).stat().st_size
        st.sidebar.success(f"🤖 Model loaded: {Path(model_path).name}")
        st.sidebar.info(f"📊 Size: {file_size / (1024*1024):.1f} MB")
        return True
    else:
        st.sidebar.warning("⚠️ Running in demo mode")
        return False

def handle_missing_model():
    """Handle the case when no model is available"""
    st.error("""
    🚫 **No Model Available**
    
    This demo requires a trained deepfake detection model. To use this app:
    
    1. **Train a model** using the provided training code
    2. **Upload the model** to cloud storage 
    3. **Set the MODEL_URL** environment variable
    4. **Restart the app**
    
    For questions, please contact the repository owner.
    """)
    
    st.stop()
