"""
Test Nemotron Parse model via /generate endpoint.

Requires a running server:
    python -m sglang.launch_server \
        --model nvidia/NVIDIA-Nemotron-Parse-v1.1 \
        --trust-remote-code \
        --port 29999

Usage:
    python test/srt/models/test_nemotron_parse.py
    python test/srt/models/test_nemotron_parse.py --port 29999
"""

import argparse
import base64
import io
import json
import re
import unittest
import urllib.request

from PIL import Image, ImageDraw

DEFAULT_PORT = 29999
DEFAULT_HOST = "127.0.0.1"

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"

SAMPLING_PARAMS = {
    "max_new_tokens": 4000,
    "temperature": 0.0,
    "repetition_penalty": 1.1,
    "top_k": 1,
}


def make_test_image():
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)

    draw.text((50, 30), "Document Title", fill="black")
    draw.text((50, 80), "This is a paragraph of text in a simple", fill="black")
    draw.text((50, 100), "document for testing OCR capabilities.", fill="black")
    draw.text((50, 160), "Section 2: Results", fill="black")
    draw.text((50, 200), "The results show improvement.", fill="black")

    for i in range(4):
        draw.line([(50, 270 + i * 30), (400, 270 + i * 30)], fill="black")
    draw.line([(50, 270), (50, 360)], fill="black")
    draw.line([(200, 270), (200, 360)], fill="black")
    draw.line([(400, 270), (400, 360)], fill="black")
    draw.text((80, 275), "Column A", fill="black")
    draw.text((260, 275), "Column B", fill="black")
    draw.text((100, 305), "1.0", fill="black")
    draw.text((290, 305), "2.0", fill="black")
    draw.text((100, 335), "3.0", fill="black")
    draw.text((290, 335), "4.0", fill="black")

    return img


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate(host, port, image_b64, prompt=PROMPT, sampling_params=None):
    if sampling_params is None:
        sampling_params = SAMPLING_PARAMS

    payload = {
        "text": prompt,
        "image_data": [f"data:image/png;base64,{image_b64}"],
        "sampling_params": sampling_params,
    }

    req = urllib.request.Request(
        f"http://{host}:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


class TestNemotronParse(unittest.TestCase):
    host = DEFAULT_HOST
    port = DEFAULT_PORT

    def test_basic_ocr(self):
        img = make_test_image()
        img_b64 = image_to_base64(img)
        result = generate(self.host, self.port, img_b64)

        text = result["text"]
        meta = result["meta_info"]

        self.assertEqual(meta["finish_reason"]["type"], "stop")
        self.assertEqual(meta["prompt_tokens"], 5)

        bbox_pattern = r"<x_[\d.]+><y_[\d.]+>"
        bboxes = re.findall(bbox_pattern, text)
        self.assertGreater(len(bboxes), 0, "No bounding boxes found in output")

        class_pattern = r"<class_[\w-]+>"
        classes = re.findall(class_pattern, text)
        self.assertGreater(len(classes), 0, "No class labels found in output")

        print(f"Found {len(bboxes)} bbox coords, {len(classes)} class labels")
        print(f"Classes: {classes}")

    def test_text_extraction(self):
        img = make_test_image()
        img_b64 = image_to_base64(img)
        result = generate(self.host, self.port, img_b64)

        text = result["text"]

        self.assertIn("Document", text)
        self.assertIn("Section", text)
        self.assertIn("Results", text)

        print(f"Extracted text (first 500 chars):\n{text[:500]}")

    def test_table_extraction(self):
        img = make_test_image()
        img_b64 = image_to_base64(img)
        result = generate(self.host, self.port, img_b64)

        text = result["text"]

        self.assertIn("<class_Table>", text)
        self.assertIn("Column A", text)
        self.assertIn("Column B", text)

        print("Table extraction OK")

    def test_class_types(self):
        img = make_test_image()
        img_b64 = image_to_base64(img)
        result = generate(self.host, self.port, img_b64)

        text = result["text"]
        classes = re.findall(r"<class_([\w-]+)>", text)

        self.assertIn("Text", classes)
        self.assertIn("Table", classes)

        print(f"Detected classes: {set(classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, remaining = parser.parse_known_args()

    TestNemotronParse.host = args.host
    TestNemotronParse.port = args.port

    unittest.main(argv=["test_nemotron_parse"] + remaining)
