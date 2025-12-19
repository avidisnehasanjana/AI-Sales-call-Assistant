import streamlit as st
import tempfile
import numpy as np
import subprocess
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from datetime import timedelta
import whisper
import os

# ---------------------------
# NLP MODELS (LOCAL)
# ---------------------------
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
sentiment_model = pipeline("sentiment-analysis")
intent_model = pipeline("zero-shot-classification")

# ----------------------------------------------------
# OLLAMA (LOCAL PHI-3 MODEL)
# ----------------------------------------------------
def ollama_chat(prompt):
    """
    Sends a prompt to local PHI-3 model via Ollama.
    """
    try:
        process = subprocess.Popen(
            ["ollama", "run", "phi3"],   # <-- USING PHI-3 MODEL
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output, error = process.communicate(prompt)

        if error:
            return f"Ollama Error: {error}"

        return output.strip()

    except FileNotFoundError:
        return "❌ ERROR: Ollama is not installed or not in PATH."


# ----------------------------------------------------
# SALES SUGGESTION ENGINE USING PHI-3
# ----------------------------------------------------
def phi3_sales_suggestions(sentence, sentiment, intent, entities):
    prompt = f"""
You are an expert sales assistant.

Customer says: "{sentence}"
Sentiment: {sentiment}
Intent: {intent}
Entities: {entities}

Your tasks:
1. Ask the next best question.
2. Provide a soft objection-handling response.
3. Recommend a suitable product.
4. Produce professional sales message with 4 lines only.

Format your response exactly like this:

Next Question:
<your text>

Objection Handling:
<your text>

Product Recommendation:
<your text>


"""

    return ollama_chat(prompt)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🎙️ AI Sales Intelligence ")
st.write("Fully offline transcription, NLP & AI sales suggestions using PHI-3.")


@st.cache_resource
def load_whisper():
    return whisper.load_model("small")


@st.cache_resource
def load_encoder():
    return VoiceEncoder()


whisper_model = load_whisper()
encoder = load_encoder()


# ---------------------------
# AUDIO INPUT
# ---------------------------
st.header("🎤 Record Audio")
recorded_audio = st.audio_input("Click to record")

st.header("📤 Upload Audio File")
uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

audio_source = recorded_audio or uploaded_audio


# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------
if audio_source and st.button("Start Analysis"):
    with st.spinner("Processing audio..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_source.read())
            temp_audio_path = tmp.name

        # Whisper transcription
        result = whisper_model.transcribe(temp_audio_path)
        segments = result["segments"]

        # Speaker diarization
        audio = AudioSegment.from_file(temp_audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        embeddings = []
        chunk_files = []

        for seg in segments:
            start, end = seg["start"], seg["end"]

            chunk_audio = audio[start * 1000:end * 1000]
            chunk_path = f"chunk_{start:.2f}.wav"
            chunk_audio.export(chunk_path, format="wav")

            chunk_files.append(chunk_path)

            wav = preprocess_wav(chunk_path)
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        # clustering
        if len(embeddings) < 2:
            labels = np.array([0])
            num_speakers = 1
        else:
            clustering = AgglomerativeClustering(
                distance_threshold=1.0, n_clusters=None
            ).fit(embeddings)

            labels = clustering.labels_
            num_speakers = len(set(labels))

        speaker_map = {i: f"Speaker {chr(65 + i)}" for i in range(num_speakers)}

        st.subheader("📝 Speaker-Labeled Transcript")

        # -------------------------
        # PROCESS EACH SENTENCE
        # -------------------------
        for i, seg in enumerate(segments):
            timestamp = str(timedelta(seconds=int(seg["start"])))
            speaker = speaker_map[labels[i]]
            text = seg["text"].strip()

            st.write(f"[{timestamp}] **{speaker}:** {text}")

            if speaker == "Speaker A":  # CUSTOMER SPEECH
                # NLP
                sentiment = sentiment_model(text)[0]["label"]
                intent = intent_model(text, [
                    "discount request",
                    "refund request",
                    "product inquiry",
                    "price negotiation",
                    "complaint",
                    "technical issue",
                    "payment issue",
                    "order status",
                    "greeting",
                    "general question"
                ])["labels"][0]

                doc = nlp(text)
                entities = {ent.label_: ent.text for ent in doc.ents}

                st.markdown("### 🔍 Customer Analysis")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Intent:** {intent}")
                st.write(f"**Entities:** {entities if entities else 'None'}")

                # ---------------------------
                # PHI-3 SUGGESTIONS
                # ---------------------------
                st.markdown("### 🤖 AI Sales Suggestions ")
                response = phi3_sales_suggestions(text, sentiment, intent, entities)
                st.write(response)

                st.markdown("---")

        # Cleanup
        for f in chunk_files:
            os.remove(f)
