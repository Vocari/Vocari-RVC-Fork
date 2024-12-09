import os
from audio_separator.separator import Separator
import subprocess
import re


def audio_separator(url, output):

    # Extract video ID from the YouTube URL
    try:
        # Using regex for more robust video ID extraction
        match = re.search(r"(?:v=|youtu\.be/)([^&?/]+)", URL)
        video_id = match.group(1) if match else None
    except Exception as e:
        print(f"Error extracting video ID: {e}")
        video_id = None

    if video_id:
        # Construct the yt-dlp command
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format",
            "wav",  # Specify WAV format
            "--output",
            f"{video_id}/{video_id}.wav",  # Set output filename
            url,  # YouTube URL
        ]

        try:
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Downloaded audio as {video_id}.wav")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading audio: {e}")
        except FileNotFoundError:
            print("yt-dlp not found.")
    else:
        print("Invalid YouTube URL")
        return

    separator = Separator(output_dir=output)

    input_vocals = f"{video_id}/{video_id}.wav"
    # Vocals and Instrumental
    vocals = os.path.join(output, "Vocals.wav")
    instrumental = os.path.join(output, "Instrumental.wav")

    # Splitting a track into Vocal and Instrumental
    separator.load_model("model_bs_roformer_ep_368_sdr_12.9628.ckpt")
    voc_inst = separator.separate(input_vocals)
    os.rename(
        os.path.join(output, voc_inst[0]), instrumental
    )  # Rename file to “Instrumental.wav”
    os.rename(os.path.join(output, voc_inst[1]), vocals)  # Rename file to “Vocals.wav”

    return instrumental, vocals
