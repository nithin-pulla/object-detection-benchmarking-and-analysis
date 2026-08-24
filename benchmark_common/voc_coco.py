"""Shared VOC -> COCO conversion (P0 fix: this was duplicated verbatim
between the FasterRCNN and SSD300 notebooks; now used by both, plus by
YOLOv5's unified-evaluation cell to build a matching ground-truth file).
"""
import json
import logging
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime
from pathlib import Path

from PIL import Image

from benchmark_common import protocol

_module_logger = logging.getLogger(__name__)


def voc_image_id_to_int(image_id):
    """Deterministic VOC-id -> integer COCO image_id.

    VOC ids look like "2008_000002", which is not a valid int() directly.
    The original per-notebook implementation fell back to Python's
    built-in hash() % 10**8 in that case, which is NOT stable across
    process runs (string-hash randomization) unless PYTHONHASHSEED is
    fixed *before* the interpreter starts. That makes it unsafe to
    reconstruct the same id later (e.g. when matching YOLOv5's
    separately-generated predictions against this ground truth). This
    version is deterministic: strip the underscore and parse as an int,
    falling back to a stable CRC32 checksum only for ids that don't fit
    that pattern.
    """
    stripped = image_id.replace("_", "")
    if stripped.isdigit():
        return int(stripped)
    return zlib.crc32(image_id.encode("utf-8"))


def convert_voc_to_coco(voc_root, output_file, image_set="trainval",
                         voc_classes=None, logger=None):
    """Convert a Pascal VOC ImageSet split to a COCO-format detection JSON.

    Returns a stats dict on success, or False if the image_set file is
    missing (matches the original notebooks' return-value contract).
    """
    voc_classes = voc_classes or protocol.VOC_CLASSES
    class_to_idx = {cls: idx for idx, cls in enumerate(voc_classes)}
    log = logger or _module_logger

    log.info(f"Converting VOC to COCO format for {image_set} set...")
    voc_path = Path(voc_root)

    coco_format = {
        "info": {
            "description": "Pascal VOC 2012 in COCO format",
            "version": "1.0",
            "year": 2012,
            "contributor": "benchmark_common.voc_coco",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{
            "id": 1,
            "name": "Pascal VOC License",
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/",
        }],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    for idx, class_name in enumerate(voc_classes):
        coco_format["categories"].append({
            "id": idx + 1,  # COCO categories start from 1
            "name": class_name,
            "supercategory": "object",
        })

    image_set_file = voc_path / "ImageSets" / "Main" / f"{image_set}.txt"
    if not image_set_file.exists():
        log.error(f"Image set file not found: {image_set_file}")
        return False

    with open(image_set_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    annotation_id = 1
    conversion_stats = {"total_images": 0, "total_annotations": 0, "skipped_images": 0}

    log.info(f"Processing {len(image_ids)} images...")

    for idx, image_id in enumerate(image_ids):
        if idx % 500 == 0:
            log.info(f"Processed {idx}/{len(image_ids)} images")

        img_file = voc_path / "JPEGImages" / f"{image_id}.jpg"
        if not img_file.exists():
            log.warning(f"Image file not found: {img_file}")
            conversion_stats["skipped_images"] += 1
            continue

        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception as e:
            log.warning(f"Cannot read image {img_file}: {e}")
            conversion_stats["skipped_images"] += 1
            continue

        img_id_int = voc_image_id_to_int(image_id)

        coco_format["images"].append({
            "id": img_id_int,
            "file_name": f"{image_id}.jpg",
            "width": width,
            "height": height,
            "license": 1,
        })
        conversion_stats["total_images"] += 1

        xml_file = voc_path / "Annotations" / f"{image_id}.xml"
        if not xml_file.exists():
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in class_to_idx:
                    continue

                bbox_elem = obj.find("bndbox")
                xmin = float(bbox_elem.find("xmin").text) - 1  # convert to 0-based
                ymin = float(bbox_elem.find("ymin").text) - 1
                xmax = float(bbox_elem.find("xmax").text)
                ymax = float(bbox_elem.find("ymax").text)

                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id_int,
                    "category_id": class_to_idx[class_name] + 1,
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0,
                })
                annotation_id += 1
                conversion_stats["total_annotations"] += 1

        except Exception as e:
            log.warning(f"Error processing annotations for {image_id}: {e}")

    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=2)

    log.info("VOC to COCO conversion completed!")
    log.info(f"Statistics: {conversion_stats}")
    log.info(f"COCO file saved to: {output_file}")

    return conversion_stats
