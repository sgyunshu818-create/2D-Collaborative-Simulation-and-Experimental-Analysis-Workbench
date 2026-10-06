"""Deterministic originals and perturbations, grouped before any dev/test use."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from . import ExperimentError, GENERATOR_VERSION, LABELS


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel_hash(image):
    return sha256(image.tobytes(order="C"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def read_image(path):
    path = Path(path)
    try:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error) as error:
        raise ExperimentError(f"{path}: cannot read PNG: {error}") from error
    if image is None:
        raise ExperimentError(f"{path}: invalid PNG image")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise ExperimentError(f"{path}: cannot encode PNG")
    content = encoded.tobytes()
    path.write_bytes(content)
    return sha256(content)


def rng_for(seed):
    # Explicit PCG64 instead of relying on a future default_rng algorithm.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def render_original(label, size, seed):
    """Return pixels, foreground mask and background; no algorithm uses masks."""
    if label not in LABELS:
        raise ValueError(f"unknown synthetic marker {label!r}")
    rng = rng_for(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    background = (rng.uniform(165, 235) + rng.uniform(-15, 15) * (xx / size - 0.5)
                  + rng.uniform(-15, 15) * (yy / size - 0.5)
                  + rng.normal(0, 1.5, (size, size)))
    background = np.clip(np.rint(background), 0, 255).astype(np.uint8)
    center = (int(rng.uniform(0.42, 0.58) * size), int(rng.uniform(0.42, 0.58) * size))
    radius = int(rng.uniform(0.20, 0.28) * size)
    angle = rng.uniform(0, 2 * math.pi)
    intensity = int(rng.integers(25, 90))
    mask = np.zeros((size, size), dtype=np.uint8)
    if label == "circle":
        cv2.circle(mask, center, radius, 255, cv2.FILLED, cv2.LINE_AA)
    else:
        vertices = 4 if label == "square" else 3
        angles = angle + np.arange(vertices) * (2 * math.pi / vertices)
        points = np.rint(np.column_stack((center[0] + radius * np.cos(angles),
                                         center[1] + radius * np.sin(angles)))).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 255, lineType=cv2.LINE_AA)
    alpha = mask.astype(np.float64) / 255
    image = np.rint(background * (1 - alpha) + intensity * alpha).astype(np.uint8)
    return image, mask, background


def perturb(image, mask, background, condition, strength, seed):
    rng = rng_for(seed)
    if condition == "normal":
        return image.copy(), 0.0
    if condition.startswith("noise_"):
        output = image.astype(np.float64) + rng.normal(0, strength, image.shape)
        return np.clip(np.rint(output), 0, 255).astype(np.uint8), 0.0
    if condition.startswith("occlusion_"):
        # A randomly oriented half-plane replaces a precise fraction of marker
        # pixels with background. Strength means hidden foreground area, not
        # fraction of the whole image; antialiased mask >=128 defines that area.
        yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        angle = rng.uniform(0, 2 * math.pi)
        projection = xx * math.cos(angle) + yy * math.sin(angle)
        foreground = mask >= 128
        values = np.sort(projection[foreground])
        hidden_count = max(1, int(round(len(values) * strength)))
        threshold = values[-hidden_count]
        occluder = projection >= threshold
        output = image.copy()
        output[occluder] = background[occluder]
        return output, float(np.count_nonzero(occluder & foreground) / len(values))
    raise ValueError(f"unknown condition {condition!r}")


def generate_dataset(config, directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ExperimentError(f"{directory}: generation requires an empty output directory; use a new output path")
    directory.mkdir(parents=True, exist_ok=True)
    originals, rows = [], []
    for split, seed, count in (("dev", config.dev_seed, config.dev_per_class),
                               ("test", config.test_seed, config.test_per_class)):
        for label_index, label in enumerate(LABELS):
            for index in range(count):
                original_id = f"{split}_{label}_{index:04d}"
                original_seed = [seed, label_index, index, 0]
                original, mask, background = render_original(label, config.image_size, original_seed)
                original_path = f"images/{split}/originals/{original_id}.png"
                original_png_hash = write_image(directory / original_path, original)
                original_pixel_hash = pixel_hash(original)
                originals.append({"original_id": original_id, "split": split, "label": label,
                                  "seed": original_seed, "path": original_path,
                                  "png_sha256": original_png_hash, "pixel_sha256": original_pixel_hash})
                for condition_index, (condition, strength) in enumerate(config.conditions):
                    variant_seed = [seed, label_index, index, condition_index + 1]
                    pixels, actual_occlusion = perturb(original, mask, background, condition,
                                                       strength, variant_seed)
                    image_id = original_id + "--" + condition
                    relative_path = f"images/{split}/{condition}/{original_id}.png"
                    png_hash = write_image(directory / relative_path, pixels)
                    rows.append({"image_id": image_id, "original_id": original_id,
                                 "split": split, "label": label, "condition": condition,
                                 "strength": strength, "actual_occlusion": actual_occlusion,
                                 "seed": variant_seed, "path": relative_path,
                                 "png_sha256": png_hash, "pixel_sha256": pixel_hash(pixels),
                                 "original_png_sha256": original_png_hash,
                                 "original_pixel_sha256": original_pixel_hash})
    manifest = {"format_version": 1, "generator_version": GENERATOR_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "rng": "NumPy PCG64 + SeedSequence (recorded integer entropy)",
                "config": config.data(), "config_sha256": config.sha256,
                "labels": LABELS, "conditions": config.conditions,
                "originals": originals, "images": rows}
    validate_manifest(manifest, directory, config, verify_images=False)
    write_json(directory / "manifest.json", manifest)
    with (directory / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {"seed": json.dumps(row["seed"])})
    return manifest


def safe_path(directory, relative):
    if not isinstance(relative, str):
        raise ExperimentError("manifest image path must be a string")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("..", ".") or ":" in part or "\\" in part for part in path.parts):
        raise ExperimentError(f"manifest path escapes its output directory: {relative!r}")
    resolved = (Path(directory) / relative).resolve()
    if not resolved.is_relative_to(Path(directory).resolve()):
        raise ExperimentError(f"manifest path escapes its output directory: {relative!r}")
    return resolved


def validate_manifest(manifest, directory, config, verify_images=True):
    if not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int or manifest["format_version"] != 1:
        raise ExperimentError(f"{directory}: unsupported dataset manifest")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ExperimentError(f"{directory}: unsupported image generator version")
    if manifest.get("config_sha256") != config.sha256:
        raise ExperimentError(f"{directory}: dataset configuration hash does not match")
    expected_data = json.loads(json.dumps(config.data()))
    if manifest.get("config") != expected_data and manifest.get("config") != config.data():
        raise ExperimentError(f"{directory}: dataset configuration was modified")
    originals, images = manifest.get("originals"), manifest.get("images")
    if not isinstance(originals, list) or not isinstance(images, list) or not originals or not images:
        raise ExperimentError(f"{directory}: manifest must contain originals and images")
    groups, image_ids, base_ids, hashes = defaultdict(list), set(), {}, {"dev": set(), "test": set()}
    original_hashes = set()
    original_counts = Counter()
    conditions = dict(config.conditions)
    for base in originals:
        if not isinstance(base, dict):
            raise ExperimentError(f"{directory}: invalid original row")
        original_id, split, label = base.get("original_id"), base.get("split"), base.get("label")
        if not isinstance(original_id, str) or original_id in base_ids or not isinstance(split, str) or split not in hashes or label not in LABELS:
            raise ExperimentError(f"{directory}: invalid or duplicate original identity")
        seed = base.get("seed")
        expected_seed = config.dev_seed if split == "dev" else config.test_seed
        count = config.dev_per_class if split == "dev" else config.test_per_class
        if (not isinstance(seed, list) or len(seed) != 4 or any(type(part) is not int for part in seed)
                or seed[0] != expected_seed or seed[1] != LABELS.index(label) or not 0 <= seed[2] < count or seed[3] != 0
                or original_id != f"{split}_{label}_{seed[2]:04d}"):
            raise ExperimentError(f"{directory}: original seed and identity do not match the split configuration")
        base_ids[original_id] = base
        original_counts[(split, label)] += 1
        for field in ("png_sha256", "pixel_sha256"):
            if not isinstance(base.get(field), str) or len(base[field]) != 64:
                raise ExperimentError(f"{directory}: invalid original hash")
        hashes[split].add(base["pixel_sha256"])
        if base["pixel_sha256"] in original_hashes:
            raise ExperimentError(f"{directory}: original images must have distinct pixel hashes")
        original_hashes.add(base["pixel_sha256"])
        if verify_images:
            _verify_image(directory, base, config.image_size)
    for row in images:
        if not isinstance(row, dict):
            raise ExperimentError(f"{directory}: invalid image row")
        original_id = row.get("original_id")
        base = base_ids.get(original_id) if isinstance(original_id, str) else None
        image_id, split, label = row.get("image_id"), row.get("split"), row.get("label")
        if not isinstance(image_id, str) or image_id in image_ids or base is None or not isinstance(split, str) or split not in hashes or label not in LABELS:
            raise ExperimentError(f"{directory}: unknown or duplicate image identity")
        image_ids.add(image_id)
        if split != base["split"] or label != base["label"]:
            raise ExperimentError(f"{directory}: all variants of an original must stay in the same split and class")
        condition = row.get("condition")
        if not isinstance(condition, str) or condition not in conditions or row.get("strength") != conditions[condition]:
            raise ExperimentError(f"{directory}: unknown image condition or altered strength")
        expected_variant_seed = base["seed"][:3] + [list(conditions).index(condition) + 1]
        if row.get("seed") != expected_variant_seed or image_id != original_id + "--" + condition:
            raise ExperimentError(f"{directory}: variant seed or identity was modified")
        if row.get("original_pixel_sha256") != base["pixel_sha256"] or row.get("original_png_sha256") != base["png_sha256"]:
            raise ExperimentError(f"{directory}: original image hash association was modified")
        for field in ("png_sha256", "pixel_sha256"):
            if not isinstance(row.get(field), str) or len(row[field]) != 64:
                raise ExperimentError(f"{directory}: invalid image hash")
        hashes[split].add(row["pixel_sha256"])
        groups[row["original_id"]].append(condition)
        if verify_images:
            _verify_image(directory, row, config.image_size)
    if hashes["dev"] & hashes["test"]:
        raise ExperimentError(f"{directory}: dev/test pixel hashes overlap")
    for split, count in (("dev", config.dev_per_class), ("test", config.test_per_class)):
        for label in LABELS:
            if original_counts[(split, label)] != count:
                raise ExperimentError(f"{directory}: incorrect {split}/{label} original count")
    if set(groups) != set(base_ids) or any(Counter(group) != Counter(conditions.keys()) for group in groups.values()):
        raise ExperimentError(f"{directory}: every original must have exactly one image for each condition")
    return manifest


def _verify_image(directory, row, size):
    path = safe_path(directory, row.get("path"))
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ExperimentError(f"{path}: cannot read dataset image: {error}") from error
    if sha256(content) != row["png_sha256"]:
        raise ExperimentError(f"{path}: PNG hash mismatch")
    image = read_image(path)
    if image.shape != (size, size) or pixel_hash(image) != row["pixel_sha256"]:
        raise ExperimentError(f"{path}: pixel hash or image dimensions mismatch")


def load_dataset(directory, config):
    path = Path(directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise ExperimentError(f"{path}: cannot read dataset manifest: {error}") from error
    return validate_manifest(manifest, directory, config)
