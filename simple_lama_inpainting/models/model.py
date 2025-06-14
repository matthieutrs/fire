import os
import torch
import numpy as np
from PIL import Image
from simple_lama_inpainting.utils.util import prepare_img_and_mask, download_model

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",  # noqa
)


class SimpleLama:
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            cur_res = Image.fromarray(cur_res)
            return cur_res


class LamaRestorer(torch.nn.Module):
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 model_path='big-lama.pt'):
        super(LamaRestorer, self).__init__()
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        # else:
        #     model_path = download_model(LAMA_MODEL_URL)
        # self.model_path = model_path
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def forward(self, image, mask=None, physics=None):
        # check if mask type is tensor
        if not torch.is_tensor(mask):
            mask = physics.mask
        return self.model(image, 1-mask)  # here we need to check, maybe the best would be to do 1-physics.mask above to have the 2 behaviours


