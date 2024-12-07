from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pyngrok import ngrok

# Create necessary directories
os.makedirs("dataset", exist_ok=True)
model_library = CachedModels()

# Initialize Tkinter app
root = tk.Tk()
root.title("Neo's RVC GUI")

# Global variables
selected_voice_model = tk.StringVar(value="")
selected_audio_file = tk.StringVar(value="")
pitch_value = tk.DoubleVar(value=0)
speaker_id = tk.IntVar(value=0)
output_path = tk.StringVar(value="")
index_strength = tk.DoubleVar(value=0.5)
method_value = tk.StringVar(value="rmvpe")

# Functions
def refresh_models():
    """Refresh available voice models and audio files."""
    voice_models = sorted(names) if names else []
    audio_files = [os.path.abspath(os.path.join('audios', f)) for f in os.listdir('audios') if
                   os.path.splitext(f)[1].lower() in ('.mp3', '.wav', '.flac', '.ogg')]
    voice_dropdown["values"] = voice_models
    audio_dropdown["values"] = audio_files

def convert_audio():
    """Handle the audio conversion process."""
    try:
        vc_output, vc_info = vc.vc_single(
            speaker_id.get(),
            selected_audio_file.get(),
            pitch_value.get(),
            None,  # F0 Path not implemented in this example
            method_value.get(),
            "",
            "",
            index_strength.get(),
            3,  # Default filter radius
            0,  # Default resample rate
            0,  # Default RMS mix rate
            0.33,  # Default breathiness protection
        )
        output_path.set(vc_output)
        messagebox.showinfo("Conversion Complete", "Audio conversion completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def open_audio_file():
    """Open a file dialog to select an audio file."""
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg")])
    if file_path:
        selected_audio_file.set(file_path)

def start_ngrok():
    """Start Ngrok and print the public URL."""
    url = ngrok.connect(5000).public_url
    print(f"Ngrok public URL: {url}")
    messagebox.showinfo("Ngrok URL", f"Access your app at {url}")

# Layout
main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Voice Model Section
ttk.Label(main_frame, text="Voice Model:").grid(row=0, column=0, sticky=tk.W)
voice_dropdown = ttk.Combobox(main_frame, textvariable=selected_voice_model)
voice_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E))
refresh_button = ttk.Button(main_frame, text="Refresh Models", command=refresh_models)
refresh_button.grid(row=0, column=2, sticky=tk.W)

# Audio Input Section
ttk.Label(main_frame, text="Input Audio File:").grid(row=1, column=0, sticky=tk.W)
audio_dropdown = ttk.Combobox(main_frame, textvariable=selected_audio_file)
audio_dropdown.grid(row=1, column=1, sticky=(tk.W, tk.E))
browse_button = ttk.Button(main_frame, text="Browse", command=open_audio_file)
browse_button.grid(row=1, column=2, sticky=tk.W)

# Conversion Parameters
ttk.Label(main_frame, text="Pitch:").grid(row=2, column=0, sticky=tk.W)
ttk.Scale(main_frame, variable=pitch_value, from_=-12, to=12, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E))
ttk.Label(main_frame, text="Speaker ID:").grid(row=3, column=0, sticky=tk.W)
speaker_entry = ttk.Entry(main_frame, textvariable=speaker_id)
speaker_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

# Conversion Button
convert_button = ttk.Button(main_frame, text="Convert Audio", command=convert_audio)
convert_button.grid(row=4, column=0, columnspan=3, pady=10)

# Output Section
ttk.Label(main_frame, text="Output Path:").grid(row=5, column=0, sticky=tk.W)
output_entry = ttk.Entry(main_frame, textvariable=output_path, state="readonly")
output_entry.grid(row=5, column=1, sticky=(tk.W, tk.E))

# Start Ngrok Button
ngrok_button = ttk.Button(main_frame, text="Start Ngrok", command=lambda: threading.Thread(target=start_ngrok).start())
ngrok_button.grid(row=6, column=0, columnspan=3, pady=10)

# Adjust column weights
main_frame.columnconfigure(1, weight=1)

# Start the application
refresh_models()
root.mainloop()
