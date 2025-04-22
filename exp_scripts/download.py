"""Download datasets and pretrained models from AIME-NoB paper."""

import os
from argparse import ArgumentParser

from aime_nob.utils import DATA_PATH, MODEL_PATH

supported_dataset_names = [
    "cartpole-plan2explore-buffer",
    "walker-plan2explore-buffer",
    "hopper-plan2explore-buffer",
    "cheetah-plan2explore-buffer",
    "finger-plan2explore-buffer",
    "quadruped-plan2explore-buffer",
    "tdmpc2-metaworld-expert-datasets",
    "patchail-dmc-expert-datasets",
    "tdmpc2-metaworld-mt39",
    "tdmpc2-metaworld-mt50",
]
supported_model_names = [
    "dmc-models",
    "metaworld-models",
]

parts = {
    "walker-plan2explore-buffer": 2,
    "hopper-plan2explore-buffer": 2,
    "cheetah-plan2explore-buffer": 2,
    "quadruped-plan2explore-buffer": 4,
    "tdmpc2-metaworld-mt50" : 2,
}

remote_dataset_url = "https://huggingface.co/datasets/IcarusWizard/AIME-NoB/resolve/main/"
remote_model_url = "https://huggingface.co/IcarusWizard/AIME-NoB/resolve/main/"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", type=str, required=True)
    parser.add_argument("--keep_files", "-k", action="store_true")
    args = parser.parse_args()

    assert (
        args.name in supported_dataset_names or args.name in supported_model_names
    ), f"please selected one of the following names: {supported_dataset_names + supported_model_names}."

    output_folder = MODEL_PATH if "model" in args.name else DATA_PATH

    print("Downloading files ...")
    files = (
        [f"{args.name}.zip"]
        if args.name not in parts.keys()
        else [f"{args.name}-part{i}.zip" for i in range(parts[args.name])]
    )
    for file in files:
        remote_url = remote_dataset_url if args.name in supported_dataset_names else remote_model_url
        os.system(f"wget -P {output_folder} {remote_url+file} --no-check-certificate")

    print("Extracting files ...")
    extract_folder = (
        output_folder
        if "buffer" not in args.name and "mt" not in args.name
        else os.path.join(output_folder, args.name)
    )
    for file in files:
        os.system(f"unzip {os.path.join(output_folder, file)} -d {extract_folder}")

    if not args.keep_files:
        print("Deleting downloaded files ...")
        for file in files:
            os.system(f"rm {os.path.join(output_folder, file)}")