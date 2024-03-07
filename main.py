import os
from typing import Optional
import torch

import requests

from wav2lip import inference
from wav2lip.models import Wav2Lip


DEIVCE = "cuda"
WEIGHTS_LINK = (
    "https://drive.usercontent.google.com/download?id=1_kKragQGB-ZHJpoQP02coX2"
    + "oL5pOJxT9&export=download&authuser=1&confirm=t&uuid=526d4a80-31b1-4d71-a4"
    + "f7-e1d676e3cbfa&at=APZUnTWE8-GrHJPQWCCoK950BRrJ%3A1708105918119"
)
WEIGHTS_NAME = "wav2lip_gan.pth"
AUDIO_PATH = "data/record_cut.wav"
VIDEO_PATH = "data/english_video_cut.mp4"
# VIDEO_PATH = "data/input_img.jpeg"
# VIDEO_PATH = "data/ez_english_cut2.mp4"
OUTPUT_PATH = "output.mp4"


def load_model(path: Optional[str] = None):
    """Load the model from the given path."""
    if not path or not os.path.exists(path):
        response = requests.get(WEIGHTS_LINK, timeout=200)
        with open(WEIGHTS_NAME, mode="wb") as file:
            file.write(response.content)

        path = WEIGHTS_NAME

    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v

    model.load_state_dict(new_s)
    model = model.to(DEIVCE)
    print("Model loaded")
    return model.eval()


def main():
    """Main script for running wav2lip."""

    model = load_model("wav2lip_gan.pth")

    inference.main(VIDEO_PATH, AUDIO_PATH, model)


if __name__ == "__main__":
    main()
