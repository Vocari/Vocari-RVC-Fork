from mega import Mega
import os
import subprocess
import shutil
import argparse


def download_from_url(url=None, model=None):
    if not url:
        print("Error: URL is required.")
        return ""

    if not model:
        try:
            model = url.split("/")[-1].split("?")[0]
        except:
            print("Error: Failed to derive model name from URL.")
            return

    model = model.replace(".pth", "").replace(".index", "").replace(".zip", "")
    url = url.replace("/blob/main/", "/resolve/main/").strip()

    for directory in ["downloads", "unzips", "zip"]:
        os.makedirs(directory, exist_ok=True)

    try:
        if url.endswith(".pth"):
            subprocess.run(["wget", url, "-O", f"assets/weights/{model}.pth"])
        elif url.endswith(".index"):
            os.makedirs(f"logs/{model}", exist_ok=True)
            subprocess.run(["wget", url, "-O", f"logs/{model}/added_{model}.index"])
        elif url.endswith(".zip"):
            subprocess.run(["wget", url, "-O", f"downloads/{model}.zip"])
        else:
            if "drive.google.com" in url:
                url = url.split("/")[0]
                subprocess.run(["gdown", url, "--fuzzy", "-O", f"downloads/{model}"])
            elif "mega.nz" in url:
                Mega().download_url(url, "downloads")
            else:
                subprocess.run(["wget", url, "-O", f"downloads/{model}"])

        downloaded_file = next((f for f in os.listdir("downloads")), None)
        if downloaded_file:
            if downloaded_file.endswith(".zip"):
                shutil.unpack_archive(f"downloads/{downloaded_file}", "unzips", "zip")
                for root, _, files in os.walk("unzips"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".index"):
                            os.makedirs(f"logs/{model}", exist_ok=True)
                            shutil.copy2(file_path, f"logs/{model}")
                        elif (
                            file.endswith(".pth")
                            and "G_" not in file
                            and "D_" not in file
                        ):
                            shutil.copy(file_path, f"assets/weights/{model}.pth")
            elif downloaded_file.endswith(".pth"):
                shutil.copy(
                    f"downloads/{downloaded_file}", f"assets/weights/{model}.pth"
                )
            elif downloaded_file.endswith(".index"):
                os.makedirs(f"logs/{model}", exist_ok=True)
                shutil.copy(
                    f"downloads/{downloaded_file}", f"logs/{model}/added_{model}.index"
                )
            else:
                print("Error: Failed to process downloaded file.")
                return "Failed"

        print("Download completed successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        shutil.rmtree("downloads", ignore_errors=True)
        shutil.rmtree("unzips", ignore_errors=True)
        shutil.rmtree("zip", ignore_errors=True)
        return "Done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from a given URL.")
    parser.add_argument(
        "--url", type=str, required=True, help="URL to download the file from."
    )
    parser.add_argument("--model", type=str, required=False, help="Name of the model.")

    args = parser.parse_args()
    download_from_url(url=args.url, model=args.model)
