from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/Vocari/VoiceConversion/resolve/main/"
BASE_DIR = Path.cwd()  # Using Path for better path management
CHUNK_SIZE = 8192  # Configurable chunk size for downloads


def dl_model(link: str, model_name: str, dir_name: Path) -> None:
    """
    Downloads a file from the given link and saves it in the specified directory.

    Args:
        link (str): Base URL of the file to download.
        model_name (str): Name of the file to download.
        dir_name (Path): Path to the directory where the file will be saved.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    try:
        response = requests.get(f"{link}{model_name}", stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        dir_name.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        file_path = dir_name / model_name
        with file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        print(f"Downloaded {model_name} to {file_path}")
    except requests.RequestException as e:
        print(f"Error downloading {model_name}: {e}")


if __name__ == "__main__":
    models = [
        ("hubert_base.pt", BASE_DIR / "assets/hubert"),
        ("rmvpe.pt", BASE_DIR / "assets/rmvpe"),
        ("fcpe.pt", BASE_DIR / "assets/fcpe"),
    ]

    for model_name, model_dir in models:
        print(f"Downloading {model_name}...")
        dl_model(RVC_DOWNLOAD_LINK, model_name, model_dir)

    print("All models downloaded!")
