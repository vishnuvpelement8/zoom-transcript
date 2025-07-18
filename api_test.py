#!/usr/bin/env python3
"""
Simple test script for the Vercel-optimized API
"""

import requests
import os

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_transcription(audio_file_path):
    """Test the transcription endpoint"""
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {"file": f}
            data = {"language": "en"}
            
            response = requests.post(
                "http://localhost:8000/transcribe",
                files=files,
                data=data
            )
            
        print(f"Transcription: {response.status_code}")
        
        if response.status_code == 200:
            with open("test_transcript.docx", "wb") as f:
                f.write(response.content)
            print("Transcript saved as test_transcript.docx")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Transcription test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Vercel-optimized API...")
    
    # Test health check
    if test_health_check():
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")
    
    # Test transcription (you'll need to provide an audio file)
    audio_files = ["sample.m4a", "sample.wav", "sample.mp3"]
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            print(f"\nTesting with {audio_file}...")
            if test_transcription(audio_file):
                print("✅ Transcription test passed")
                break
            else:
                print("❌ Transcription test failed")
    else:
        print("\nNo test audio files found. Create one of:")
        for audio_file in audio_files:
            print(f"  - {audio_file}")
