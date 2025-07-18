#!/usr/bin/env python3
"""
Create a simple test audio file for testing the transcript API
"""

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os

def create_test_audio():
    """Create a simple test audio file with some tones"""
    print("🎵 Creating test audio file...")
    
    # Create a simple audio sequence with different tones
    # This will create a 10-second audio file with different frequencies
    
    # Generate different tones (frequencies in Hz)
    tone1 = Sine(440).to_audio_segment(duration=2000)  # A note for 2 seconds
    tone2 = Sine(523).to_audio_segment(duration=2000)  # C note for 2 seconds
    tone3 = Sine(659).to_audio_segment(duration=2000)  # E note for 2 seconds
    
    # Add some silence between tones
    silence = AudioSegment.silent(duration=500)  # 0.5 seconds of silence
    
    # Combine the tones with silence
    test_audio = tone1 + silence + tone2 + silence + tone3 + silence
    
    # Reduce volume to make it more pleasant
    test_audio = test_audio - 20  # Reduce by 20dB
    
    # Export as different formats
    formats = [
        ("test_audio.wav", "wav"),
        ("test_audio.mp3", "mp3"),
        ("test_audio.m4a", "mp4")  # m4a uses mp4 container
    ]
    
    for filename, format_name in formats:
        try:
            test_audio.export(filename, format=format_name)
            file_size = os.path.getsize(filename) / 1024  # Size in KB
            print(f"✅ Created {filename} ({file_size:.1f} KB)")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    print("\n📝 Note: These are just tone files, not speech.")
    print("   The transcript will likely be empty or contain artifacts.")
    print("   For real testing, use actual speech audio files.")

if __name__ == "__main__":
    try:
        create_test_audio()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install pydub")
    except Exception as e:
        print(f"❌ Error creating test audio: {e}")
