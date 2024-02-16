from typing import Optional
import torch
from wav2lip import inference
from wav2lip.models import Wav2Lip
import gdown


DEIVCE = "cpu"
AUDIO_PATH = "record.wav"
VIDEO_PATH = "vid.mp4"
OUTPUT_PATH = "output.mp4"


def load_model(path: Optional[str] = None):
    wav2lip_checkpoints_url = "https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing"
    if not path:
        gdown.download_folder(wav2lip_checkpoints_url, quiet=True, use_cookies=False)
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    model = model.to(DEIVCE)
    return model.eval()


def main():

    model = load_model()

    inference.main(VIDEO_PATH, AUDIO_PATH, model)


if __name__ == "__main__":
    main()
