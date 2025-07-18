#!/usr/bin/env python3
"""
FastAPI Zoom Meeting Transcript Generator
Upload audio files via POST request to get transcripts
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import whisper
from docx import Document
from datetime import datetime
import os
import re
import tempfile
import shutil
from pydub import AudioSegment
import uuid
from pathlib import Path
from typing import Optional

app = FastAPI(
    title="Zoom Meeting Transcript API",
    description="Upload audio files to get meeting transcripts",
    version="1.0.0"
)

# Configuration
TARGET_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = -40
MIN_SILENCE_DURATION = 1000
CHUNK_SIZE = 300
MAX_AUDIO_LENGTH = 1800
UPLOAD_DIR = "uploads"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global model variable (loaded once)
whisper_model = None

@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup"""
    global whisper_model
    print("🔄 Loading Whisper model...")
    try:
        # Use tiny model for Vercel deployment to reduce package size
        model_name = os.getenv("WHISPER_MODEL_NAME", "tiny")
        whisper_model = whisper.load_model(model_name)
        print(f"✅ Whisper model '{model_name}' loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Zoom Meeting Transcript API",
        "status": "running",
        "model_loaded": whisper_model is not None
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...),language: Optional[str] = "en"):
    """
    Upload an audio file and get the transcript file directly

    - **file**: Audio file (m4a, mp4, wav, mp3, etc.)
    - **language**: Language code (default: en)

    Returns the transcript as a downloadable Word document
    """
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    # Validate file type
    allowed_extensions = {'.m4a', '.mp4', '.wav', '.mp3', '.flac', '.ogg'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique filename
    unique_id = str(uuid.uuid4())
    temp_audio_path = os.path.join(UPLOAD_DIR, f"{unique_id}{file_extension}")
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name

    try:
        # Save uploaded file
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📁 Processing file: {file.filename}")

        # Preprocess audio
        processed_audio = preprocess_audio(temp_audio_path)

        # Transcribe
        segments = transcribe_fast(whisper_model, processed_audio, language)

        # Assign speakers
        speaker_segments = assign_speakers_fast(segments)

        # Create transcript
        create_fast_transcript(speaker_segments, file.filename, output_path)

        # Clean up temp files (but keep output_path for FileResponse)
        cleanup_temp_files([temp_audio_path, processed_audio])

        # Return the transcript file directly
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Transcript file was not created")

        # Generate a clean filename based on original file
        original_name = Path(file.filename).stem
        clean_filename = f"transcript_{original_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

        # FileResponse will handle cleanup of the output file after sending
        return FileResponse(
            path=output_path,
            filename=clean_filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            background=BackgroundTask(cleanup_temp_files, [output_path])  # Clean up after sending
        )

    except Exception as e:
        # Clean up on error
        cleanup_temp_files([temp_audio_path])
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def preprocess_audio(audio_path):
    """Preprocess audio for optimal Whisper performance"""
    print("🔧 Preprocessing audio...")
    
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        print(f"   Duration: {duration_seconds:.1f} seconds")
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to 16kHz
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        
        # Normalize and trim silence
        audio = audio.normalize()
        audio = audio.strip_silence(silence_len=1000, silence_thresh=SILENCE_THRESHOLD)
        
        # Handle long audio
        if duration_seconds > MAX_AUDIO_LENGTH:
            return split_audio_into_chunks(audio)
        
        # Save optimized audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_file.name, format='wav')
        return temp_file.name
        
    except Exception as e:
        print(f"⚠️ Audio preprocessing failed: {e}")
        return audio_path

def split_audio_into_chunks(audio):
    """Split long audio into manageable chunks"""
    chunks = []
    chunk_length_ms = CHUNK_SIZE * 1000
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        chunk.export(temp_file.name, format='wav')
        chunks.append(temp_file.name)
    
    print(f"   Split into {len(chunks)} chunks")
    return chunks

def transcribe_fast(model, audio_path, language="en"):
    """Fast transcription with optimized settings"""
    print("🎵 Transcribing...")
    
    config = {
        "language": language,
        "task": "transcribe",
        "verbose": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "beam_size": 1,
        "best_of": 1,
        "fp16": True,
    }
    
    if isinstance(audio_path, list):
        return transcribe_chunks(model, audio_path, config)
    else:
        result = model.transcribe(audio_path, **config)
        return process_transcription_result(result, 0)

def transcribe_chunks(model, audio_chunks, config):
    """Transcribe multiple audio chunks"""
    all_segments = []
    time_offset = 0
    
    for i, chunk_path in enumerate(audio_chunks):
        print(f"   Processing chunk {i+1}/{len(audio_chunks)}...")
        result = model.transcribe(chunk_path, **config)
        segments = process_transcription_result(result, time_offset)
        all_segments.extend(segments)
        time_offset += CHUNK_SIZE
        
        try:
            os.unlink(chunk_path)
        except:
            pass
    
    return all_segments

def process_transcription_result(result, time_offset=0):
    """Process and clean transcription results"""
    segments = result.get("segments", [])
    clean_segments = []
    
    for segment in segments:
        text = clean_transcription_text(segment["text"])
        
        if not is_meaningful_text(text):
            continue
        
        if segment.get("avg_logprob", 0) < -1.5:
            continue
        
        clean_segments.append({
            "start": segment["start"] + time_offset,
            "end": segment["end"] + time_offset,
            "text": text,
            "confidence": segment.get("avg_logprob", 0)
        })
    
    return clean_segments

def clean_transcription_text(text):
    """Clean transcription text"""
    if not text:
        return ""
    
    text = re.sub(r'\b(nd|uh|um|ah)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_meaningful_text(text):
    """Check for meaningful content"""
    if not text or len(text.strip()) < 3:
        return False
    
    words = text.split()
    if len(words) < 1:
        return False
    
    text_lower = text.lower()
    artifacts = ['odi recorder', 'install whisper', 'pip install']
    if any(artifact in text_lower for artifact in artifacts):
        return False
    
    return True

def assign_speakers_fast(segments):
    """Improved speaker assignment for natural conversation flow"""
    if not segments:
        return []

    speaker_segments = []
    current_speaker = 1
    last_end_time = 0

    for i, segment in enumerate(segments):
        text = segment["text"].strip().lower()
        pause_duration = segment["start"] - last_end_time

        should_switch = False

        # Long pause suggests speaker change
        if pause_duration > 2.0:
            should_switch = True

        # Response indicators
        response_words = ["yes", "no", "yeah", "okay", "right", "well", "so", "but", "and", "oh"]
        if any(text.startswith(word + " ") or text == word for word in response_words):
            should_switch = True

        # Question-answer pattern
        if i > 0:
            prev_text = segments[i-1]["text"].strip()
            if prev_text.endswith("?"):
                should_switch = True

        if should_switch:
            current_speaker = 2 if current_speaker == 1 else 1

        speaker_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": f"Speaker {current_speaker}",
            "confidence": segment["confidence"],
            "prev_end": last_end_time
        })

        last_end_time = segment["end"]

    return speaker_segments

def create_fast_transcript(segments, original_filename, output_path):
    """Create transcript with proper conversation flow"""
    print("📄 Creating transcript...")

    doc = Document()
    doc.add_heading("Meeting Transcript", 0)
    doc.add_paragraph(f"Audio File: {original_filename}")
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")

    if not segments:
        doc.add_paragraph("No meaningful conversation detected.")
        doc.save(output_path)
        return

    doc.add_heading("Conversation", 1)

    conversation_turns = []
    current_speaker = None
    current_text = ""
    current_start_time = 0

    for seg in segments:
        text = seg["text"].strip()
        speaker = seg["speaker"]

        if speaker != current_speaker:
            if current_speaker and current_text:
                conversation_turns.append({
                    "speaker": current_speaker,
                    "text": current_text.strip(),
                    "start_time": current_start_time
                })

            current_speaker = speaker
            current_text = text
            current_start_time = seg["start"]
        else:
            time_gap = seg["start"] - seg.get("prev_end", seg["start"])

            if time_gap > 5.0 and current_text:
                conversation_turns.append({
                    "speaker": current_speaker,
                    "text": current_text.strip(),
                    "start_time": current_start_time
                })
                current_text = text
                current_start_time = seg["start"]
            else:
                if current_text and not current_text.endswith(('.', '!', '?')):
                    current_text += " " + text
                else:
                    current_text += " " + text

    # Add final turn
    if current_speaker and current_text:
        conversation_turns.append({
            "speaker": current_speaker,
            "text": current_text.strip(),
            "start_time": current_start_time
        })

    # Write conversation
    for turn in conversation_turns:
        text = turn["text"]

        if text and not text[-1] in '.!?':
            text += "."

        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        minutes = int(turn["start_time"] // 60)
        seconds = int(turn["start_time"] % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"

        doc.add_paragraph(f"{timestamp} {turn['speaker']}: {text}")

    # Summary
    doc.add_paragraph("")
    doc.add_heading("Summary", 1)

    speaker_stats = {}
    for turn in conversation_turns:
        speaker = turn["speaker"]
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"turns": 0, "words": 0}
        speaker_stats[speaker]["turns"] += 1
        speaker_stats[speaker]["words"] += len(turn["text"].split())

    doc.add_paragraph(f"Total conversation turns: {len(conversation_turns)}")
    for speaker, stats in speaker_stats.items():
        doc.add_paragraph(f"{speaker}: {stats['turns']} turns, {stats['words']} words")

    doc.save(output_path)
    print(f"✅ Transcript saved: {output_path}")

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for path in file_paths:
        try:
            if path and os.path.exists(path) and ('tmp' in path or 'uploads' in path or path.endswith('.docx')):
                os.unlink(path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
