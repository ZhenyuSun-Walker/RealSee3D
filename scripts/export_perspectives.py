"""Export Realsee3D panoramas in the TD_mansion RGB/depth/camera layout.

Each scene receives rgb/0.png, depth/0.exr, cameras/0.json, ..., 99.*.
World coordinates remain in the original Realsee3D metric frame. OpenCV
cameras use X right, Y down, Z forward; Blender cameras use X right, Y up,
Z backward. EXR channel V stores float32 optical-axis Z in meters, with 0
for missing depth. RGB alpha records panorama validity, independently of depth.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

# Limit numerical libraries before importing them, including in spawned workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import Imath
import numpy as np
import OpenEXR
from scipy.spatial.transform import Rotation


VERSION = 1
CV_TO_BLENDER = np.diag([1., -1., -1., 1.])
REQUIRED = ("panoImage_1600.jpg", "depth_image.png", "depth_scale.txt", "extrinsics.txt")


def write_json(path, value):
    """Only expose a complete JSON file, also when a run is interrupted."""
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def write_depth(path, depth):
    height, width = depth.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"V": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    pending = path.with_name(f".{path.stem}.{os.getpid()}.exr")
    image = OpenEXR.OutputFile(str(pending), header)
    try:
        image.writePixels({"V": np.ascontiguousarray(depth, dtype=np.float32).tobytes()})
    finally:
        image.close()
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def read_pose(path):
    pose = np.loadtxt(path).reshape(4, 4)
    if (not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise ValueError(f"Not a rigid camera-to-world pose: {path}")
    return pose


def view_rotation(yaw_deg, pitch_deg):
    """Perspective-to-panorama rotation; positive pitch looks upward."""
    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]]) @ np.array(
        [[1., 0., 0.], [0., cp, -sp], [0., sp, cp]])


def pinhole_rays(config):
    width, height = config["width"], config["height"]
    fx = config["lens_mm"] * width / config["sensor_width_mm"]
    k = np.array([[fx, 0., width / 2.], [0., fx, height / 2.], [0., 0., 1.]])
    v, u = np.indices((height, width), dtype=np.float32)
    rays = np.stack([(u - k[0, 2]) / fx, (v - k[1, 2]) / fx, np.ones_like(u)], axis=-1)
    inverse_norm = 1. / np.linalg.norm(rays, axis=-1)
    return k, rays, inverse_norm


def panorama_maps(rays, rotation, width, height):
    # Same pixel-to-sphere convention as scripts/generate_pcd.py; no half-pixel offset.
    x = rays[..., 0] * rotation[0, 0] + rays[..., 1] * rotation[0, 1] + rotation[0, 2]
    y = rays[..., 0] * rotation[1, 0] + rays[..., 1] * rotation[1, 1] + rotation[1, 2]
    z = rays[..., 0] * rotation[2, 0] + rays[..., 1] * rotation[2, 1] + rotation[2, 2]
    u = np.remainder((np.arctan2(x, z) / (2 * np.pi) + .5) * width, width)
    v = (np.arctan2(y, np.hypot(x, z)) / np.pi + .5) * height
    return u.astype(np.float32), np.clip(v, 0, height - 1).astype(np.float32)


def farthest_positions(poses, count):
    """Deterministic spatial coverage, including when there are >100 panoramas."""
    points = np.array([pose[:3, 3] for pose in poses])
    first = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    selected = []
    distances = np.full(len(points), np.inf)
    index = first
    for _ in range(count):
        selected.append(index)
        distances = np.minimum(distances, np.sum((points - points[index]) ** 2, axis=1))
        distances[selected] = -1.
        index = int(np.argmax(distances))
    return selected


def plan_views(scene, config):
    registry = scene / "viewpoints.txt"
    ids = registry.read_text().split() if registry.is_file() else sorted(
        p.name for p in (scene / "viewpoints").iterdir() if p.is_dir())
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate viewpoint IDs: {registry}")
    available, poses, missing = [], [], []
    for viewpoint in ids:
        directory = scene / "viewpoints" / viewpoint
        if not all((directory / name).is_file() for name in REQUIRED):
            missing.append(viewpoint)
            continue
        poses.append(read_pose(directory / "extrinsics.txt"))
        available.append(viewpoint)
    if not available:
        raise ValueError(f"No complete RGB-D viewpoints: {scene}")
    order = farthest_positions(poses, min(len(available), config["count"]))
    base, remainder = divmod(config["count"], len(order))
    frames = []
    for rank, index in enumerate(order):
        n = base + (rank < remainder)
        # Even coverage within each panorama; stagger the starting yaw across positions.
        offset = (rank * 137.50776405003785) % 360.
        for direction in range(n):
            frames.append({"id": len(frames), "viewpoint_id": available[index],
                           "yaw_degrees": (offset + 360. * direction / n) % 360.,
                           "pitch_degrees": config["pitch_degrees"]})
    return frames, dict(zip(available, poses)), missing


def load_capture(directory):
    rgb = cv2.imread(str(directory / "panoImage_1600.jpg"), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(directory / "depth_image.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"Unreadable RGB / uint16 depth: {directory}")
    scale = float((directory / "depth_scale.txt").read_text())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid depth scale: {directory}")
    mask_path = directory / "pano_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.is_file() else None
    if mask_path.is_file() and mask is None:
        raise ValueError(f"Unreadable panorama mask: {mask_path}")
    if mask is None:
        mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask == 255, 255, 0).astype(np.uint8)
    floor_path = directory / "floor.txt"
    floor = floor_path.read_text().strip() if floor_path.is_file() else None
    return rgb, raw, scale, mask, floor


def project_capture(capture, rays, inverse_norm, rotation, config):
    rgb, raw, scale, mask, _ = capture
    rgb_h, rgb_w = rgb.shape[:2]
    map_u, map_v = panorama_maps(rays, rotation, rgb_w, rgb_h)
    color = cv2.remap(rgb, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # Require the entire RGB interpolation footprint to be valid at blind-spot boundaries.
    alpha = cv2.remap(mask.astype(np.float32), map_u, map_v, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_WRAP) >= 254.999
    color[~alpha] = 0
    rgba = np.dstack([color, alpha.astype(np.uint8) * 255])
    dh, dw = raw.shape
    if (dh, dw) == (rgb_h, rgb_w):
        depth_u, depth_v = map_u, map_v
    else:
        depth_u, depth_v = panorama_maps(rays, rotation, dw, dh)
    # Nearest sampling preserves sparse LiDAR holes and avoids blending across surfaces.
    distances = cv2.remap(raw, depth_u, depth_v, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_WRAP).astype(np.float32) / scale
    z = distances * inverse_norm
    valid = alpha & np.isfinite(z) & (z >= config["clip_near"]) & (z <= config["clip_far"])
    z[~valid] = 0.
    return rgba, z


def direction_key(viewpoint_id, yaw, pitch):
    return viewpoint_id, round(yaw % 360., 5), round(pitch, 5)


def project_valid_view(scene, frame, poses, capture, rays, inverse_norm, config, reserved):
    """Recover an empty view with measured depth; never synthesize depth values."""
    rotation = view_rotation(frame["yaw_degrees"], frame["pitch_degrees"])
    rgba, depth = project_capture(capture, rays, inverse_norm, rotation, config)
    if np.any(depth > 0):
        return frame, capture, rotation, rgba, depth
    source_id = frame["viewpoint_id"]
    center = poses[source_id][:3, 3]
    others = sorted((key for key in poses if key != source_id),
                    key=lambda key: float(np.sum((poses[key][:3, 3] - center) ** 2)))
    step = max(1, min(rays.shape[:2]) // 64)
    small_rays, small_inverse_norm = rays[::step, ::step], inverse_norm[::step, ::step]
    offsets = [15. * i for i in range(1, 24)] + [7.5 + 15. * i for i in range(24)]
    for candidate_id in [source_id, *others]:
        candidate_capture = capture if candidate_id == source_id else load_capture(
            scene / "viewpoints" / candidate_id)
        ranked = []
        for offset in offsets:
            yaw = (frame["yaw_degrees"] + offset) % 360.
            if direction_key(candidate_id, yaw, frame["pitch_degrees"]) in reserved:
                continue
            candidate_rotation = view_rotation(yaw, frame["pitch_degrees"])
            _, small_depth = project_capture(candidate_capture, small_rays, small_inverse_norm,
                                              candidate_rotation, config)
            fraction = float(np.mean(small_depth > 0))
            if fraction >= .01:
                ranked.append((fraction, offset, yaw, candidate_rotation))
        for _, _, yaw, candidate_rotation in sorted(ranked, key=lambda item: (-item[0], item[1])):
            rgba, depth = project_capture(candidate_capture, rays, inverse_norm, candidate_rotation, config)
            if np.mean(depth > 0) < .01:
                continue
            actual = {**frame, "viewpoint_id": candidate_id, "yaw_degrees": yaw,
                      "requested_source_viewpoint_id": source_id,
                      "requested_yaw_degrees": frame["yaw_degrees"],
                      "sampling_fallback_reason": "requested_direction_has_no_valid_clipped_depth"}
            return actual, candidate_capture, candidate_rotation, rgba, depth
    raise ValueError(f"No usable measured depth for perspective frame {frame['id']} after searching {len(poses)} viewpoints")


def camera_record(scene, frame, pose, rotation, k, capture, config, rgba, depth):
    camera_to_world_cv = pose.copy()
    camera_to_world_cv[:3, :3] = pose[:3, :3] @ rotation
    world_to_camera_cv = np.linalg.inv(camera_to_world_cv)
    camera_to_world_blender = camera_to_world_cv @ CV_TO_BLENDER
    world_to_camera_blender = CV_TO_BLENDER @ world_to_camera_cv
    width, height = config["width"], config["height"]
    near, far = config["clip_near"], config["clip_far"]
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = 2 * k[0, 0] / width, 2 * k[1, 1] / height
    projection[2, 2], projection[2, 3] = -(far + near) / (far - near), -2 * far * near / (far - near)
    projection[3, 2] = -1.
    return {
        "name": str(frame["id"]),
        "image_path": f"{scene.name}/rgb/{frame['id']}.png",
        "depth_path": f"{scene.name}/depth/{frame['id']}.exr",
        "resolution": [width, height], "projection_type": "PERSP",
        "intrinsic_matrix": k.tolist(),
        "world_to_camera_blender": world_to_camera_blender.tolist(),
        "camera_to_world_blender": camera_to_world_blender.tolist(),
        "world_to_camera_cv": world_to_camera_cv.tolist(),
        "camera_to_world_cv": camera_to_world_cv.tolist(),
        "projection_matrix": projection.tolist(),
        "camera_location": camera_to_world_cv[:3, 3].tolist(),
        "camera_rotation_euler": Rotation.from_matrix(camera_to_world_blender[:3, :3]).as_euler("xyz").tolist(),
        "lens_mm": config["lens_mm"], "sensor_width_mm": config["sensor_width_mm"],
        "sensor_height_mm": config["sensor_height_mm"], "shift": [0., 0.],
        "clip": [near, far],
        "source_view_name": f"panorama_{frame['viewpoint_id']}_yaw_{frame['yaw_degrees']:.6f}",
        "source_viewpoint_id": frame["viewpoint_id"],
        "source_panorama_path": f"viewpoints/{frame['viewpoint_id']}/panoImage_1600.jpg",
        "source_depth_scale": capture[2], "floor": capture[4],
        "yaw_degrees": frame["yaw_degrees"], "pitch_degrees": frame["pitch_degrees"],
        "depth_type": "camera_z", "depth_units": "meters", "depth_channel": "V",
        "depth_background": 0., "depth_sampling": "nearest_no_hole_filling",
        "valid_depth_fraction": float(np.mean(depth > 0)),
        "valid_rgb_fraction": float(np.mean(rgba[..., 3] > 0)),
        "world_coordinate_system": "original_realsee3d_world_y_down",
        "rgb_alpha": "255=valid_panorama_pixel,0=invalid",
        "export_config": config,
        **{key: frame[key] for key in ("requested_source_viewpoint_id", "requested_yaw_degrees",
                                      "sampling_fallback_reason") if key in frame},
    }


@contextmanager
def scene_lock(state):
    state.mkdir(exist_ok=True)
    with (state / "lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another exporter holds the scene lock: {state.parent}")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def frame_paths(scene, frame_id):
    return (scene / "rgb" / f"{frame_id}.png", scene / "depth" / f"{frame_id}.exr",
            scene / "cameras" / f"{frame_id}.json")


def frame_exists(scene, frame_id):
    return all(p.is_file() and p.stat().st_size > 0 for p in frame_paths(scene, frame_id))


def export_scene(scene, config, force=False):
    cv2.setNumThreads(1)
    scene = Path(scene)
    state = scene / ".perspective_export"
    started = time.monotonic()
    with scene_lock(state):
        old_config = json.loads((state / "config.json").read_text()) if (state / "config.json").is_file() else None
        if old_config != config:
            # Existing unrelated RGB-D files are never overwritten by adopting a directory.
            occupied = any((scene / folder).is_dir() and any((scene / folder).iterdir())
                           for folder in ("rgb", "depth", "cameras"))
            if occupied and (old_config is None or not force
                             or old_config["count"] != config["count"]):
                raise ValueError(f"Existing RGB/depth/cameras have a different or unknown export configuration: {scene}")
        if not force and old_config == config and (state / "complete.json").is_file():
            if all(frame_exists(scene, index) for index in range(config["count"])):
                return {"scene": str(scene), "status": "existing", "seconds": time.monotonic() - started}
        (state / "complete.json").unlink(missing_ok=True)
        for folder in ("rgb", "depth", "cameras"):
            (scene / folder).mkdir(exist_ok=True)
        write_json(state / "config.json", config)
        plan, poses, missing = plan_views(scene, config)
        manifest = {"config": config, "scene": scene.name,
                    "source_viewpoints_used": len(set(f["viewpoint_id"] for f in plan)),
                    "incomplete_source_viewpoints": missing, "frames": plan,
                    "world_transform": "identity; original Realsee3D world coordinates"}
        manifest_path = state / "manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            if previous["frames"] != plan and not force:
                raise ValueError(f"Source viewpoint plan changed since the previous export: {scene}")
        write_json(manifest_path, manifest)
        k, rays, inverse_norm = pinhole_rays(config)
        current_id, capture = None, None
        depth_validity, rgb_validity = [], []
        reserved = {direction_key(f["viewpoint_id"], f["yaw_degrees"], f["pitch_degrees"]) for f in plan}
        if not force:
            for path in (scene / "cameras").glob("*.json"):
                try:
                    saved = json.loads(path.read_text())
                    if saved.get("export_config") == config:
                        reserved.add(direction_key(saved["source_viewpoint_id"], saved["yaw_degrees"], saved["pitch_degrees"]))
                except (ValueError, KeyError, OSError):
                    continue
        used_sources, fallback_count = set(), 0
        with (state / "export.log").open("a", buffering=1) as log:
            log.write(f"Start {datetime.now(timezone.utc).isoformat()}\n")
            if missing:
                log.write(f"Incomplete source viewpoints excluded: {missing}\n")
            for frame in plan:
                paths = frame_paths(scene, frame["id"])
                if not force and frame_exists(scene, frame["id"]):
                    try:
                        record = json.loads(paths[2].read_text())
                    except (ValueError, OSError):
                        record = {}
                    if record.get("export_config") == config:
                        depth_validity.append(record["valid_depth_fraction"])
                        rgb_validity.append(record["valid_rgb_fraction"])
                        used_sources.add(record["source_viewpoint_id"])
                        fallback_count += "sampling_fallback_reason" in record
                        reserved.add(direction_key(record["source_viewpoint_id"], record["yaw_degrees"], record["pitch_degrees"]))
                        continue
                if frame["viewpoint_id"] != current_id:
                    current_id = frame["viewpoint_id"]
                    capture = load_capture(scene / "viewpoints" / current_id)
                actual, actual_capture, rotation, rgba, depth = project_valid_view(
                    scene, frame, poses, capture, rays, inverse_norm, config, reserved)
                record = camera_record(scene, actual, poses[actual["viewpoint_id"]], rotation, k,
                                       actual_capture, config, rgba, depth)
                # Camera JSON is the per-frame commit marker. Delete it before replacement.
                paths[2].unlink(missing_ok=True)
                pending_rgb = paths[0].with_name(f".{frame['id']}.{os.getpid()}.png")
                try:
                    if not cv2.imwrite(str(pending_rgb), rgba, [cv2.IMWRITE_PNG_COMPRESSION, config["png_compression"]]):
                        raise OSError(f"Could not write PNG: {pending_rgb}")
                    pending_rgb.replace(paths[0])
                finally:
                    pending_rgb.unlink(missing_ok=True)
                write_depth(paths[1], depth)
                write_json(paths[2], record)
                depth_validity.append(record["valid_depth_fraction"])
                rgb_validity.append(record["valid_rgb_fraction"])
                used_sources.add(actual["viewpoint_id"])
                fallback_count += "sampling_fallback_reason" in actual
                reserved.add(direction_key(actual["viewpoint_id"], actual["yaw_degrees"], actual["pitch_degrees"]))
                log.write(f"{frame['id']}: source={actual['viewpoint_id']} yaw={actual['yaw_degrees']:.6f} "
                          f"valid_depth={record['valid_depth_fraction']:.6f} "
                          f"fallback={'sampling_fallback_reason' in actual}\n")
            expected = {str(i) for i in range(config["count"])}
            for folder, suffix in (("rgb", ".png"), ("depth", ".exr"), ("cameras", ".json")):
                found = {p.stem for p in (scene / folder).glob(f"*{suffix}") if not p.name.startswith(".")}
                if found != expected:
                    raise ValueError(f"Unexpected/missing numbered outputs in {scene / folder}")
            output_bytes = sum(p.stat().st_size for i in range(config["count"]) for p in frame_paths(scene, i))
            result = {"scene": str(scene), "status": "completed", "frames": len(plan),
                      "source_viewpoints_used": len(used_sources), "fallback_frames": fallback_count,
                      "mean_valid_depth_fraction": float(np.mean(depth_validity)),
                      "mean_valid_rgb_fraction": float(np.mean(rgb_validity)),
                      "output_bytes": output_bytes, "seconds": time.monotonic() - started,
                      "completed_utc": datetime.now(timezone.utc).isoformat()}
            write_json(state / "complete.json", result)
            log.write(json.dumps(result) + "\n")
            return result


def run_scene(scene, config, force):
    try:
        return export_scene(scene, config, force)
    except Exception as error:
        result = {"scene": str(scene), "status": "failed", "error": str(error),
                  "traceback": traceback.format_exc()}
        state = Path(scene) / ".perspective_export"
        if state.is_dir():
            # Do not replace another worker's state; errors also appear in the batch log.
            with (state / "errors.log").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="Export one scene in place.")
    source.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RealSee3D")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--lens-mm", type=float, default=22.)
    parser.add_argument("--sensor-width-mm", type=float, default=36.)
    parser.add_argument("--sensor-height-mm", type=float, default=24.)
    parser.add_argument("--pitch-degrees", type=float, default=0.)
    parser.add_argument("--clip-near", type=float, default=.03)
    parser.add_argument("--clip-far", type=float, default=200.)
    parser.add_argument("--png-compression", type=int, choices=range(10), default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Limit the number of scenes.")
    parser.add_argument("--force", action="store_true", help="Re-export owned files, allowing resolution changes with the same frame count.")
    parser.add_argument("--min-free-gb", type=float, default=50., help="Stop submitting scenes below this free disk space.")
    parser.add_argument("--summary", type=Path, help="Save a batch completion report at this path.")
    args = parser.parse_args()
    if min(args.count, args.width, args.height, args.workers) <= 0:
        parser.error("count, width, height, and workers must be positive")
    values = [args.lens_mm, args.sensor_width_mm, args.sensor_height_mm, args.clip_near, args.clip_far]
    if not all(np.isfinite(v) and v > 0 for v in values) or args.clip_far <= args.clip_near:
        parser.error("Lens, sensor, and clip values must be positive and finite; far must exceed near")
    if not np.isfinite(args.pitch_degrees) or abs(args.pitch_degrees) >= 90:
        parser.error("pitch-degrees must be finite and between -90 and 90")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if not np.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error("min-free-gb must be nonnegative and finite")
    config = {"version": VERSION, **{key: getattr(args, key) for key in (
        "count", "width", "height", "lens_mm", "sensor_width_mm", "sensor_height_mm",
        "pitch_degrees", "clip_near", "clip_far", "png_compression")}}
    if args.source:
        scenes = [args.source.absolute()]
    else:
        subsets = {"all": ("real_world_data", "synthetic_data"),
                   "real": ("real_world_data",), "synthetic": ("synthetic_data",)}[args.subset]
        scenes = []
        for subset in subsets:
            directory = args.data_root / subset
            if not directory.is_dir():
                parser.error(f"Missing source subset: {directory}")
            scenes.extend(sorted((p.absolute() for p in directory.iterdir()
                                  if p.is_dir() and p.name.startswith(("scene_", "synthetic_scene_"))), key=str))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes or any(not (scene / "viewpoints").is_dir() for scene in scenes):
        parser.error("No scenes found or a scene has no viewpoints directory")
    print(json.dumps({"event": "start", "scenes": len(scenes), "config": config, "workers": args.workers}), flush=True)
    counts = {"completed": 0, "existing": 0, "failed": 0}
    failures, total_bytes, total_seconds = [], 0, 0.
    started = time.monotonic()
    stopped = None
    # Keep a bounded queue so the free-space guard also applies during long runs.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending, cursor = {}, 0
        while cursor < len(scenes) or pending:
            while stopped is None and cursor < len(scenes) and len(pending) < args.workers:
                scene = scenes[cursor]
                free = shutil.disk_usage(scene).free
                if free < args.min_free_gb * 1024 ** 3:
                    stopped = f"Free disk space below {args.min_free_gb} GiB at {scene}"
                    print(json.dumps({"event": "stopped", "reason": stopped}), flush=True)
                    break
                pending[pool.submit(run_scene, scene, config, args.force)] = scene
                cursor += 1
            if not pending:
                break
            future = next(as_completed(pending))
            scene = pending.pop(future)
            try:
                result = future.result()
            except Exception as error:
                result = {"scene": str(scene), "status": "failed", "error": str(error)}
                stopped = "Worker process failed; stopping new submissions"
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            else:
                total_bytes += result.get("output_bytes", 0)
                total_seconds += result.get("seconds", 0)
            print(json.dumps({**result, "progress": sum(counts.values()), "total": len(scenes)}), flush=True)
    report = {"config": config, "counts": counts, "total_scenes": len(scenes),
              "unprocessed": len(scenes) - sum(counts.values()), "failures": failures,
              "stopped": stopped, "output_bytes_written": total_bytes,
              "worker_seconds": total_seconds, "wall_seconds": time.monotonic() - started,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.summary, report)
    print(json.dumps({"event": "finished", **report}), flush=True)
    return 1 if failures or stopped else 0


if __name__ == "__main__":
    sys.exit(main())
