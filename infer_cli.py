####
# USAGE
#
# In your Terminal or CMD or whatever


import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.configs.config import Config
from rvc.infer.modules.vc.modules import VC


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0method", type=str, default="harvest", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="output path")
    parser.add_argument("--model_name", type=str, help="model name stored in assets/weight_root")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half precision (True or False)")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sample rate")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="RMS mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect factor")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # Reset arguments for safety

    return args


def main():
    load_dotenv()
    args = arg_parse()

    # Ensure output directory exists
    output_dir = Path(args.opt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and initialize VC
    config = Config()
    config.device = args.device if args.device else config.device
    config.is_half = args.is_half if args.is_half else config.is_half
    vc = VC(config)
    vc.get_vc(args.model_name)

    # Perform voice conversion
    _, wav_opt = vc.vc_single(
        speaker_id=0,
        input_path=args.input_path,
        f0_up_key=args.f0up_key,
        emb_path=None,
        f0_method=args.f0method,
        index_path=args.index_path,
        emb_uv=None,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )

    # Write the output wav file
    wavfile.write(args.opt_path, wav_opt[0], wav_opt[1])


if __name__ == "__main__":
    main()


