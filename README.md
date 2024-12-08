
# Vocari's RVC Fork

This repository provides a web-based interface for Vocari's fork of the Retrieval-based Voice Conversion (RVC) model. It allows users to convert voices between different speakers using state-of-the-art retrieval-based voice conversion techniques.

## Features

- **Web-based Interface:** Simple and intuitive UI to upload and convert voice samples.
- **Retrieval-based Voice Conversion:** High-quality conversion between different speakers, retaining natural prosody and speech characteristics.
- **Model Support:** Compatible with various pre-trained models.
- **Easy Training Model:** easy options to trainig models.
  
## Prerequisites

Before running the web UI, you need the following dependencies:

- Python 3.7 or higher
- [CUDA-enabled GPU](https://developer.nvidia.com/cuda-zone) (for faster processing)
- FFmpeg for audio processing
- Git (for cloning the repository)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Vocari/Vocari-RVC-Fork.git
   cd Vocari-RVC-Fork
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models (if not already included):

   Follow the instructions in the repository for acquiring models or run a script to download them automatically.

4. Start the web server:

   ```bash
   python app.py
   ```

   The web UI will be accessible at `http://localhost:5000`.


## use cli

you can also use cli by

```bash
python rvc/tools/infer_cli.py -h
```

## Usage

1. Open a browser and go to `http://localhost:5000`.
2. Upload an audio file that you want to convert.
3. Choose the target voice model for conversion.
4. Wait for the processing to complete, and download the converted file.


## Troubleshooting

- **Missing FFmpeg:** If you receive an error about missing FFmpeg, ensure it's installed and available in your system's PATH.
  
  Install FFmpeg via [FFmpeg official website](https://ffmpeg.org/download.html) or via a package manager:
  
  ```bash
  sudo apt install ffmpeg  # For Ubuntu
  ```

- **CUDA Errors:** Ensure that your GPU drivers and CUDA toolkit are correctly installed if you are using GPU acceleration. You can verify CUDA installation with:
  
  ```bash
  nvcc --version
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vocari fork for creating the Retrieval-based Voice Conversion model.
- Thanks the developers of the original RVC repository.





<a href="https://github.com/Vocari/Vocari-RVC-Fork/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=Vocari/Vocari-RVC-Fork" alt="Contributors" />
</a>
