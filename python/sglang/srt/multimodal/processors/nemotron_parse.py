import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nemotron_parse import NemotronParseForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import load_image

logger = logging.getLogger(__name__)


class NemotronParseProcessor(BaseMultimodalProcessor):
    models = [NemotronParseForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)

        # Image target size from config
        self.target_height, self.target_width = getattr(
            hf_config, "image_size", [2048, 1648]
        )

    def _resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio to fit within target dimensions."""
        width, height = image.size
        aspect_ratio = width / height

        new_height = height
        new_width = width

        if new_height > self.target_height:
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)

        if new_width > self.target_width:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)

        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.BILINEAR)

        return image

    def _pad_to_target(self, image: Image.Image) -> Image.Image:
        """Pad image with white to target size."""
        width, height = image.size
        if width == self.target_width and height == self.target_height:
            return image

        padded = Image.new(
            "RGB", (self.target_width, self.target_height), (255, 255, 255)
        )
        padded.paste(image, (0, 0))
        return padded

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor [C, H, W].

        sglang's RadioModel does not have an internal input_conditioner,
        so we apply CLIP normalization here (matching RADIO's expected input).
        """
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

        # CLIP normalization (RADIO's input_conditioner uses these values)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not image_data:
            return None

        if len(image_data) != 1:
            raise ValueError(
                f"Nemotron Parse expects exactly 1 image input, got {len(image_data)}"
            )

        # Load image
        pil_image, _ = load_image(image_data[0])
        pil_image = pil_image.convert("RGB")

        # Resize with aspect ratio, pad to target size
        pil_image = self._resize_with_aspect_ratio(pil_image)
        pil_image = self._pad_to_target(pil_image)

        # Convert to normalized tensor (ToTensor + CLIP normalization)
        pixel_values = self._image_to_tensor(pil_image)

        # Build decoder input_ids: </s><s><predict_bbox><predict_classes><output_markdown>
        eos_token_id = getattr(self.hf_config, "eos_token_id", 2)
        bos_token_id = getattr(self.hf_config, "bos_token_id", 0)

        # Resolve special token IDs
        predict_bbox_id = self._tokenizer.convert_tokens_to_ids("<predict_bbox>")
        predict_classes_id = self._tokenizer.convert_tokens_to_ids("<predict_classes>")
        output_markdown_id = self._tokenizer.convert_tokens_to_ids("<output_markdown>")

        input_ids = [
            eos_token_id,
            bos_token_id,
            predict_bbox_id,
            predict_classes_id,
            output_markdown_id,
        ]

        return {
            "input_ids": input_ids,
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_values,
                    modality=Modality.IMAGE,
                )
            ],
        }
