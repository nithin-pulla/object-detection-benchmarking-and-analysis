from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from benchmark.data.class_map import CLASS_TO_ID


@dataclass
class GroundTruthObject:
    class_id: int
    class_name: str
    bbox_xyxy: list[float]


@dataclass
class GroundTruthImage:
    image_id: str
    image_path: str
    width: int
    height: int
    objects: list[GroundTruthObject]


def load_split_ids(voc_root: str | Path, split: str) -> list[str]:
    split_file = Path(voc_root) / "ImageSets" / "Main" / f"{split}.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_annotation(xml_path: str | Path) -> tuple[int, int, list[GroundTruthObject]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)  # type: ignore[union-attr]
    height = int(size.find("height").text)  # type: ignore[union-attr]

    objects: list[GroundTruthObject] = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower().strip()  # type: ignore[union-attr]
        if class_name not in CLASS_TO_ID:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)  # type: ignore[union-attr]
        ymin = float(bbox.find("ymin").text)  # type: ignore[union-attr]
        xmax = float(bbox.find("xmax").text)  # type: ignore[union-attr]
        ymax = float(bbox.find("ymax").text)  # type: ignore[union-attr]
        objects.append(
            GroundTruthObject(
                class_id=CLASS_TO_ID[class_name],
                class_name=class_name,
                bbox_xyxy=[xmin, ymin, xmax, ymax],
            )
        )

    return width, height, objects


def load_ground_truth(voc_root: str | Path, split: str, image_ids: list[str] | None = None) -> list[GroundTruthImage]:
    voc_root = str(voc_root)
    image_ids = image_ids or load_split_ids(voc_root, split)
    items: list[GroundTruthImage] = []
    for image_id in image_ids:
        image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg")
        ann_path = os.path.join(voc_root, "Annotations", f"{image_id}.xml")
        width, height, objects = _parse_annotation(ann_path)

        if not os.path.exists(image_path):
            # fallback to png if needed
            image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image for {image_id} not found")

        items.append(
            GroundTruthImage(
                image_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
            )
        )
    return items


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")
