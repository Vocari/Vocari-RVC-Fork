# Import necessary modules
from original import *
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pyngrok import ngrok
from easyfuncs import download_from_url, CachedModels

# Create necessary directories
os.makedirs("dataset", exist_ok=True)
model_library = CachedModels()

# Initialize Tkinter app
root = tk.Tk()
root.title("Vocari RVC Fork GUI")

# Global variables
selected_voice_model = tk.StringVar(value="")
selected_audio_file = tk.StringVar(value="")
pitch_value = tk.DoubleVar(value=0)
speaker_id = tk.IntVar(value=0)
output_path = tk.StringVar(value="")
index_strength = tk.DoubleVar(value=0.5)
method_value = tk.StringVar(value="rmvpe")

url_input = tk.StringVar(value="")
name_output = tk.StringVar(value="")
selected_model_library = tk.StringVar(value="")


# Functions
def refresh_models():
    """Refresh available voice models and audio files."""
    voice_models = sorted(names) if names else []
    audio_files = [
        os.path.abspath(os.path.join("audios", f))
        for f in os.listdir("audios")
        if os.path.splitext(f)[1].lower() in (".mp3", ".wav", ".flac", ".ogg")
    ]
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
        messagebox.showinfo(
            "Conversion Complete", "Audio conversion completed successfully!"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def open_audio_file():
    """Open a file dialog to select an audio file."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg")]
    )
    if file_path:
        selected_audio_file.set(file_path)


def download_model():
    """Download a model from the provided URL."""
    try:
        download_from_url(url_input.get(), name_output.get())
        messagebox.showinfo(
            "Download Complete", f"Model '{name_output.get()}' downloaded successfully!"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def download_from_library():
    """Download a model from the model library."""
    try:
        model_url = model_library.models.get(selected_model_library.get())
        if model_url:
            download_from_url(model_url, selected_model_library.get())
            messagebox.showinfo(
                "Download Complete",
                f"Model '{selected_model_library.get()}' downloaded successfully!",
            )
        else:
            messagebox.showerror("Error", "Invalid model selected.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def start_ngrok():
    """Start Ngrok and print the public URL."""
    url = ngrok.connect(5000).public_url
    print(f"Ngrok public URL: {url}")
    messagebox.showinfo("Ngrok URL", f"Access your app at {url}")


# Layout
main_frame = ttk.Notebook(root)

# Tab 1: Conversion
conversion_tab = ttk.Frame(main_frame)
main_frame.add(conversion_tab, text="Conversion")

# Voice Model Section
ttk.Label(conversion_tab, text="Voice Model:").grid(row=0, column=0, sticky=tk.W)
voice_dropdown = ttk.Combobox(conversion_tab, textvariable=selected_voice_model)
voice_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E))
refresh_button = ttk.Button(
    conversion_tab, text="Refresh Models", command=refresh_models
)
refresh_button.grid(row=0, column=2, sticky=tk.W)

# Audio Input Section
ttk.Label(conversion_tab, text="Input Audio File:").grid(row=1, column=0, sticky=tk.W)
audio_dropdown = ttk.Combobox(conversion_tab, textvariable=selected_audio_file)
audio_dropdown.grid(row=1, column=1, sticky=(tk.W, tk.E))
browse_button = ttk.Button(conversion_tab, text="Browse", command=open_audio_file)
browse_button.grid(row=1, column=2, sticky=tk.W)

# Conversion Parameters
ttk.Label(conversion_tab, text="Pitch:").grid(row=2, column=0, sticky=tk.W)
ttk.Scale(
    conversion_tab, variable=pitch_value, from_=-12, to=12, orient=tk.HORIZONTAL
).grid(row=2, column=1, sticky=(tk.W, tk.E))
ttk.Label(conversion_tab, text="Speaker ID:").grid(row=3, column=0, sticky=tk.W)
speaker_entry = ttk.Entry(conversion_tab, textvariable=speaker_id)
speaker_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

# Conversion Button
convert_button = ttk.Button(conversion_tab, text="Convert Audio", command=convert_audio)
convert_button.grid(row=4, column=0, columnspan=3, pady=10)

# Output Section
ttk.Label(conversion_tab, text="Output Path:").grid(row=5, column=0, sticky=tk.W)
output_entry = ttk.Entry(conversion_tab, textvariable=output_path, state="readonly")
output_entry.grid(row=5, column=1, sticky=(tk.W, tk.E))

# Start Ngrok Button
ngrok_button = ttk.Button(
    conversion_tab,
    text="Start Ngrok",
    command=lambda: threading.Thread(target=start_ngrok).start(),
)
ngrok_button.grid(row=6, column=0, columnspan=3, pady=10)

# Tab 2: Download Models
download_tab = ttk.Frame(main_frame)
main_frame.add(download_tab, text="Download Models")

# URL Download Section
ttk.Label(download_tab, text="URL to model:").grid(row=0, column=0, sticky=tk.W)
url_entry = ttk.Entry(download_tab, textvariable=url_input)
url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
ttk.Label(download_tab, text="Save as:").grid(row=1, column=0, sticky=tk.W)
name_entry = ttk.Entry(download_tab, textvariable=name_output)
name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
download_button = ttk.Button(
    download_tab, text="Download Model", command=download_model
)
download_button.grid(row=1, column=2, sticky=tk.W)

# Model Browser Section
ttk.Label(download_tab, text="Search Models (Quality UNKNOWN):").grid(
    row=2, column=0, sticky=tk.W
)
model_dropdown = ttk.Combobox(
    download_tab,
    textvariable=selected_model_library,
    values=list(model_library.models.keys()),
)
model_dropdown.grid(row=2, column=1, sticky=(tk.W, tk.E))
library_button = ttk.Button(download_tab, text="Get", command=download_from_library)
library_button.grid(row=2, column=2, sticky=tk.W)

# Pack main frame
main_frame.pack(fill=tk.BOTH, expand=True)

# Start application
refresh_models()
root.mainloop()
