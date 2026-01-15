import os
import uuid
import warnings
import streamlit as st
import whisper
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from huggingface_hub import HfApi, create_repo
from audio_recorder_streamlit import audio_recorder

# ============================
# SUPPRESS FP16 WARNING
# ============================
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ============================
# HUGGING FACE SECRETS
# ============================
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_REPO_ID = st.secrets["HF_REPO_ID"]

# ============================
# PATHS
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADED_DIR = os.path.join(BASE_DIR, "audio", "uploaded")
RECORDED_DIR = os.path.join(BASE_DIR, "audio", "recorded")
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(RECORDED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================
# LOAD WHISPER MODEL
# ============================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# ============================
# HELPERS
# ============================
def save_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(UPLOADED_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path, filename, "uploaded"

def save_recorded_audio(audio_bytes):
    filename = f"{uuid.uuid4()}.wav"
    save_path = os.path.join(RECORDED_DIR, filename)
    audio_np = np.frombuffer(audio_bytes, np.int16)
    write(save_path, 44100, audio_np)
    return save_path, filename, "recorded"

def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def save_metadata(audio_type, audio_filename, transcription):
    new_row = pd.DataFrame([{
        "audio": f"audio/{audio_type}/{audio_filename}",
        "text": transcription
    }])
    if os.path.exists(METADATA_PATH):
        df = pd.read_csv(METADATA_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(METADATA_PATH, index=False)
    print("✅ Metadata saved:", METADATA_PATH)

api = HfApi()

def push_to_huggingface():
    # Ensure repo exists
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    print("✅ Repo exists or created successfully")

    audio_folder_path = os.path.join(BASE_DIR, "audio")
    metadata_path = METADATA_PATH

    # Upload audio folder
    api.upload_folder(
        folder_path=audio_folder_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    # Upload metadata.csv
    api.upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo="metadata.csv",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("✅ Upload complete")

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Voice Dataset Collector", page_icon="🎙️")
st.title("🎙️ Voice Dataset Collector")
st.write("Record or upload voice → Transcribe → Save to Hugging Face")

tab1, tab2 = st.tabs(["🎤 Record Voice", "📂 Upload Audio"])

# ============================
# RECORD TAB
# ============================
with tab1:
    st.subheader("🎤 Record your voice")

    audio_bytes = audio_recorder(
        text="Click to start recording",
        type="wav",
        key="audio_recorder"
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.success("✅ Audio recorded successfully!")

        with st.spinner("Saving recorded audio..."):
            audio_path, filename, audio_type = save_recorded_audio(audio_bytes)

        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_path)

        if transcription:
            st.markdown("### 📝 Transcription")
            st.write(transcription)
            save_metadata(audio_type, filename, transcription)

            with st.spinner("Uploading to Hugging Face..."):
                push_to_huggingface()

            st.success("✅ Recorded audio saved & uploaded!")

    else:
        st.info("🎤 Click the button above to record your voice. Allow microphone access in your browser.")

# ============================
# UPLOAD TAB
# ============================
with tab2:
    st.subheader("📂 Upload audio file")
    uploaded_file = st.file_uploader(
        "Upload WAV, MP3, M4A, or MP4",
        type=["wav", "mp3", "m4a", "mp4"]
    )
    if uploaded_file:
        with st.spinner("Saving uploaded file..."):
            audio_path, filename, audio_type = save_uploaded_file(uploaded_file)
        st.audio(audio_path)

        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_path)

        if transcription:
            st.markdown("### 📝 Transcription")
            st.write(transcription)
            save_metadata(audio_type, filename, transcription)

            with st.spinner("Uploading to Hugging Face..."):
                push_to_huggingface()

            st.success("✅ Uploaded audio saved & uploaded!")

# ============================
# DISPLAY METADATA
# ============================
if os.path.exists(METADATA_PATH):
    st.subheader("📄 Current Dataset Metadata")
    st.dataframe(pd.read_csv(METADATA_PATH))
