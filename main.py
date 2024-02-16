from typing import Optional
import torch
from wav2lip import inference
from wav2lip.models import Wav2Lip
import requests


DEIVCE = "cpu"
WEIGHTS_LINK = "https://drive.usercontent.google.com/download?id=1_kKragQGB-ZHJpoQP02coX2oL5pOJxT9&export=download&authuser=1&confirm=t&uuid=526d4a80-31b1-4d71-a4f7-e1d676e3cbfa&at=APZUnTWE8-GrHJPQWCCoK950BRrJ%3A1708105918119"
AUDIO_PATH = "/kaggle/input/audio-record/record.wav"
VIDEO_PATH = "/kaggle/input/video-dubbing/english_video.mp4"
OUTPUT_PATH = "output.mp4"


def load_model(path: Optional[str] = None):
    if not path:
        response = requests.get(WEIGHTS_LINK)
        with open("wav2lip_gan.pth", mode="wb") as file:
            file.write(response.content)

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

    model = load_model("/kaggle/working/wav2lip_gan.pth")

    inference.main(VIDEO_PATH, AUDIO_PATH, model)


# main()
