from original import *
import shutil, glob
import os, subprocess
import gradio as gr
import shutil, time, torch, gc
from mega import Mega
from datetime import datetime
import pandas as pd
import os, sys, subprocess, numpy as np
from pydub import AudioSegment
import huggingface_hub
import zipfile
import os

now_dir = os.getcwd()
sys.path.append(now_dir)

pretraineds_custom_path = os.path.join(
    now_dir, "rvc", "models", "pretraineds", "pretraineds_custom"
)


pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]


pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")


def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )


def show(path, ext, on_error=None):
    try:
        return list(filter(lambda x: x.endswith(ext), os.listdir(path)))
    except:
        return on_error


def run_subprocess(command):
    try:
        subprocess.run(command, check=True)
        return True, None
    except Exception as e:
        return False, e


def download_from_url(url=None, model=None):
    if not url:
        try:
            url = model[f"{model}"]
        except:
            gr.Warning("Failed")
            return ""
    if model == "":
        try:
            model = url.split("/")[-1].split("?")[0]
        except:
            gr.Warning("Please name the model")
            return
    model = model.replace(".pth", "").replace(".index", "").replace(".zip", "")
    url = url.replace("/blob/main/", "/resolve/main/").strip()

    for directory in ["downloads", "unzips", "zip"]:
        # shutil.rmtree(directory, ignore_errors=True)
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
                gr.Warning("Failed to download file")
                return "Failed"

        gr.Info("Done")
    except Exception as e:
        gr.Warning(f"There's been an error: {str(e)}")
    finally:
        shutil.rmtree("downloads", ignore_errors=True)
        shutil.rmtree("unzips", ignore_errors=True)
        shutil.rmtree("zip", ignore_errors=True)
        return "Done"


def upload_model(repo_id, pth, index, token):  # Changed 'repo' to 'repo_id'
    """
    Upload a model to the Hugging Face Hub

    Args:
        repo_id: str, the name of the repository # Changed 'repo' to 'repo_id'
        pth: str, path to the model file
        index: str, the index of the model in the repository
        token: str, the API token

    Returns:
        str, message indicating the success of the operation
    """

    repo_name = repo_id.split("/")[1]  # Changed 'repo' to 'repo_id'
    with zipfile.ZipFile(f"{repo_name}.zip", "w") as zipf:
        zipf.write(pth, os.path.basename(pth))
        zipf.write(index, os.path.basename(index))

    # Use repo_id instead of name in create_repo, and use 'local_path' instead of 'path' for upload_file
    huggingface_hub.HfApi().create_repo(repo_id=repo_id, token=token, exist_ok=True)
    huggingface_hub.HfApi().upload_file(
        path_or_fileobj=f'{repo_id.split("/")[1]}.zip',
        path_in_repo=f'{repo_id.split("/")[1]}.zip',  # Changed 'repo' to 'repo_id'
        repo_id=repo_id,
        token=token,
    )
    os.remove(f'{repo_id.split("/")[1]}.zip')  # Changed 'repo' to 'repo_id'
    return "Model uploaded successfully"


with gr.Blocks(
    title="🔊 Vocari's RVC Fork",
    theme=gr.themes.Base(primary_hue="sky", neutral_hue="zinc"),
) as app:
    with gr.Row():
        gr.Markdown("# Vocari's RVC Fork")
    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                with gr.Row():
                    voice_model = gr.Dropdown(
                        label="Model Voice",
                        choices=sorted(names),
                        value=lambda: (
                            sorted(names)[0] if len(sorted(names)) > 0 else ""
                        ),
                        interactive=True,
                    )
                    file_index2 = gr.Dropdown(
                        label="Change Index",
                        choices=sorted(index_paths),
                        interactive=True,
                        value=(
                            sorted(index_paths)[0]
                            if len(sorted(index_paths)) > 0
                            else ""
                        ),
                    )
                with gr.Row():      
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label="Speaker ID",
                        value=0,
                        visible=False,
                        interactive=True,
                    )
                    vc_transform0 = gr.Number(label="Pitch", value=0)
                with gr.Row():      
                    refresh_button = gr.Button("Refresh", variant="primary")
                    but0 = gr.Button(value="Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Upload"):
                            dropbox = gr.File(
                                label="Drop your audio here & hit the Reload button."
                            )
                        with gr.TabItem("Record"):
                            record_button = gr.Audio(
                                label="OR Record audio.",
                                type="filepath",
                            )

                        with gr.TabItem("Upload models after Training"):
                            voice_model1 = gr.Dropdown(
                                label="Model Files",
                                choices=sorted(names),
                                value=lambda: (
                                    sorted(names)[0] if len(sorted(names)) > 0 else ""
                                ),
                                interactive=True,
                            )
                            voice_index = gr.Dropdown(
                                label="Index Files",
                                choices=sorted(index_paths),
                                interactive=True,
                                value=(
                                    sorted(index_paths)[0]
                                    if len(sorted(index_paths)) > 0
                                    else ""
                                ),
                            )
                            with gr.Row():
                                repo_url = gr.Textbox(
                                    label="your url",
                                )
                                hf_token = gr.Textbox(
                                    label="your token",
                                )
                            upload_modelst = gr.Button(
                                value="Upload models", variant="primary"
                            )
                            upload_modelst.click(
                                fn=upload_model,
                                inputs=[repo_url, voice_model1, voice_index, hf_token],
                                outputs=[hf_token],
                            )

                    with gr.Row():
                        paths_for_files = lambda path: [
                            os.path.abspath(os.path.join(path, f))
                            for f in os.listdir(path)
                            if os.path.splitext(f)[1].lower()
                            in (".mp3", ".wav", ".flac", ".ogg")
                        ]
                        input_audio0 = gr.Dropdown(
                            label="Input Path",
                            value=(
                                paths_for_files("audios")[0]
                                if len(paths_for_files("audios")) > 0
                                else ""
                            ),
                            choices=paths_for_files(
                                "audios"
                            ),  # Only show absolute paths for audio files ending in .mp3, .wav, .flac or .ogg
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        audio_player = gr.Audio(label="Input")
                        input_audio0.change(
                            inputs=[input_audio0],
                            outputs=[audio_player],
                            fn=lambda path: (
                                {"value": path, "__type__": "update"}
                                if os.path.exists(path)
                                else None
                            ),
                        )
                        record_button.stop_recording(
                            fn=lambda audio: audio,  # TODO save wav lambda
                            inputs=[record_button],
                            outputs=[input_audio0],
                        )
                        dropbox.upload(
                            fn=lambda audio: audio.name,
                            inputs=[dropbox],
                            outputs=[input_audio0],
                        )

    
                    with gr.Accordion("General Settings", open=False):
                        f0method0 = gr.Radio(
                            label="Method",
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )

                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Index Strength",
                            value=0.5,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Breathiness Reduction (Harvest only)",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample",
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume Normalization",
                            value=0,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Breathiness Protection (0 is enabled, 0.5 is disabled)",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        if voice_model != None:
                            vc.get_vc(voice_model.value, protect0, protect0)
                    file_index1 = gr.Textbox(
                        label="Index Path",
                        interactive=True,
                        visible=False,  # Not used here
                    )
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[voice_model, file_index2],
                        api_name="infer_refresh",
                    )
                    refresh_button.click(
                        fn=lambda: {
                            "choices": paths_for_files("audios"),
                            "__type__": "update",
                        },  # TODO check if properly returns a sorted list of audio files in the 'audios' folder that have the extensions '.wav', '.mp3', '.ogg', or '.flac'
                        inputs=[],
                        outputs=[input_audio0],
                    )
                    refresh_button.click(
                        fn=lambda: (
                            {
                                "value": paths_for_files("audios")[0],
                                "__type__": "update",
                            }
                            if len(paths_for_files("audios")) > 0
                            else {"value": "", "__type__": "update"}
                        ),  # TODO check if properly returns a sorted list of audio files in the 'audios' folder that have the extensions '.wav', '.mp3', '.ogg', or '.flac'
                        inputs=[],
                        outputs=[input_audio0],
                    )
                    with gr.Row():
                        f0_file = gr.File(label="F0 Path", visible=False)

                    with gr.Row():
                        vc_output2 = gr.Audio(label="Output")
                    with gr.Row():      
                        vc_output1 = gr.Textbox(label="Information", placeholder="output here!", visible=True)
                but0.click(
                    vc.vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
                voice_model.change(
                    fn=vc.get_vc,
                    inputs=[voice_model, protect0, protect0],
                    outputs=[spk_item, protect0, protect0, file_index2, file_index2],
                    api_name="infer_change_voice",
                )
        with gr.TabItem("Download RVC Models"):
            url = gr.Textbox(label="url")
            model_name = gr.Textbox(label="Model name")
            download_md = gr.Button("Download")
            download_md.click(
                fn=download_from_url, inputs=[url, model_name], outputs=model_name
            )
        with gr.TabItem("Train"):
            with gr.Row():
                with gr.Column():
                    training_name = gr.Textbox(
                        label="Name your model:",
                        value="My-Voice",
                        placeholder="My-Voice",
                    )
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label="Number of CPU processes used to extract pitch features",
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    sr2 = gr.Radio(
                        label="Sampling Rate:",
                        choices=["40k", "32k"],
                        value="32k",
                        interactive=True,
                        visible=True,
                    )
                    if_f0_3 = gr.Radio(
                        label="Will your model be used for singing? If not, you can ignore this.",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        visible=False,
                    )
                    version19 = gr.Radio(
                        label="Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                    dataset_folder = gr.Textbox(
                        label="dataset folder:", value="dataset"
                    )

                    but1 = gr.Button("1. Process:", variant="primary")
                    info1 = gr.Textbox(label="Information:", value="", visible=True)

                    gpus6 = gr.Textbox(
                        label="Enter the GPU numbers to use separated by -, (e.g. 0-1-2):",
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpu_info9 = gr.Textbox(
                        label="GPU Info", value=gpu_info, visible=F0GPUVisible
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Speaker ID:",
                        value=0,
                        interactive=True,
                        visible=False,
                    )
                    but1.click(
                        preprocess_dataset,
                        [dataset_folder, training_name, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
                with gr.Column():
                    f0method8 = gr.Radio(
                        label="F0 extraction method:",
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label="GPU numbers to use separated by -, (e.g. 0-1-2)",
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    but2 = gr.Button("2. Extract Features:", variant="primary")
                    info2 = gr.Textbox(label="Information:", value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            training_name,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
                with gr.Column():
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label="Epochs (more epochs may improve quality but takes longer)",
                        value=150,
                        interactive=True,
                    )
                    but4 = gr.Button("3. Train Index", variant="primary")
                    but3 = gr.Button("4. Train Model", variant="primary")
                    info3 = gr.Textbox(label="Information", value="", max_lines=10)
                    with gr.Accordion(label="General Settings", open=False):
                        gpus16 = gr.Textbox(
                            label="GPUs separated by -, (e.g. 0-1-2)",
                            value="0",
                            interactive=True,
                            visible=True,
                        )
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label="Weight Saving Frequency",
                            value=25,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label="Batch Size",
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label="Only save the latest model",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                            visible=False,
                        )
                        if_cache_gpu17 = gr.Radio(
                            label="If your dataset is UNDER 10 minutes, cache it to train faster",
                            choices=["yes", "no"],
                            value="no",
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label="Save small model at every save point",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                        with gr.Accordion(label="Change pretrains", open=False):

                            pretrained_G14 = gr.Dropdown(
                                label=("Custom Pretrained G"),
                                info=(
                                    "Select the custom pretrained model for the generator."
                                ),
                                choices=sorted(pretraineds_list_g),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            pretrained_D15 = gr.Dropdown(
                                label=("Custom Pretrained D"),
                                info=(
                                    "Select the custom pretrained model for the generator."
                                ),
                                choices=sorted(pretraineds_list_d),
                                interactive=True,
                                allow_custom_value=True,
                            )
                    with gr.Row():
                        download_model = gr.Button("5.Download Model")
                    with gr.Row():
                        model_files = gr.Files(
                            label="Your Model and Index file can be downloaded here:"
                        )
                        download_model.click(
                            fn=lambda name: os.listdir(f"assets/weights/{name}")
                            + glob.glob(f'logs/{name.split(".")[0]}/added_*.index'),
                            inputs=[training_name],
                            outputs=[model_files, info3],
                        )
                    with gr.Row():
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19],
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                    with gr.Row():
                       
                        but3.click(
                            click_train,
                            [
                                training_name,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                            api_name="train_start",
                        )
                        but4.click(train_index, [training_name, version19], info3)

    if config.iscolab:
        app.queue()
        app.launch(share=True)
    else:
        app.launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
