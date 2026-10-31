"""Render a colored Realsee3D ASCII PLY as an orthographic top-down PNG.

World X points right in the image and world Z points up. World Y points down;
the smallest retained Y is the surface visible from above. Height limits remove
ceilings and points below the floor. No mesh or learned image synthesis is used.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def scene_metadata(scene_dir):
    """Return floor IDs and median camera Y when viewpoint metadata is available."""
    floors = set()
    camera_y = []
    if scene_dir is None or not (scene_dir / "viewpoints").is_dir():
        return floors, None
    for viewpoint in sorted((scene_dir / "viewpoints").iterdir()):
        if not viewpoint.is_dir():
            continue
        floor_path = viewpoint / "floor.txt"
        if floor_path.is_file():
            floors.add(floor_path.read_text().strip())
        pose_path = viewpoint / "extrinsics.txt"
        if pose_path.is_file():
            pose = np.loadtxt(pose_path).reshape(4, 4)
            camera_y.append(pose[1, 3])
    return floors, float(np.median(camera_y)) if camera_y else None


def estimate_planes(points, camera_y=None):
    """Estimate the dominant floor and ceiling Y levels for a single story.

    Extreme scan outliers are excluded. The strongest height-histogram peak on
    each side of the cameras identifies a horizontal surface, rather than using
    the raw minimum/maximum, which may be outdoor or erroneous scan points.
    """
    y = points[np.isfinite(points[:, :3]).all(axis=1), 1]
    if len(y) < 20:
        raise ValueError("Too few finite points to estimate floor and ceiling")
    low, high = np.percentile(y, [0.5, 99.5])
    if high - low < 0.5:
        raise ValueError("Insufficient vertical extent for floor/ceiling estimation")
    reference = float(np.median(y)) if camera_y is None else camera_y
    edges = np.arange(low - 0.02, high + 0.04, 0.02)
    counts, edges = np.histogram(y, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    density = np.convolve(counts, [1, 2, 3, 2, 1], mode="same")

    def peak(mask):
        candidates = np.flatnonzero(mask)
        if not len(candidates) or counts[candidates].sum() == 0:
            raise ValueError("Could not find a plane on both sides of the cameras")
        center = centers[candidates[np.argmax(density[candidates])]]
        return float(np.median(y[np.abs(y - center) <= 0.04]))

    ceiling_y = peak(centers < reference - 0.2)
    floor_y = peak(centers > reference + 0.2)
    if floor_y - ceiling_y < 0.5:
        raise ValueError("Estimated floor and ceiling are too close")
    return floor_y, ceiling_y


def read_ply(path):
    with path.open() as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Input is not a PLY file")
        properties = []
        vertex_count = None
        in_vertices = False
        ascii_format = False
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "format":
                ascii_format = fields[1] == "ascii"
            elif fields[0] == "element":
                in_vertices = fields[1] == "vertex"
                if in_vertices:
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and in_vertices:
                if fields[1] == "list":
                    raise ValueError("List properties on vertices are unsupported")
                properties.append(fields[-1])
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError("Missing PLY end_header")
        if not ascii_format or not vertex_count:
            raise ValueError("Expected a nonempty ASCII PLY")
        columns = [properties.index(name) for name in
                   ("x", "y", "z", "red", "green", "blue")]
        points = np.loadtxt(stream, dtype=np.float32, usecols=columns,
                            max_rows=vertex_count, ndmin=2)
        if len(points) != vertex_count:
            raise ValueError("PLY vertex count does not match the file contents")
        return points


def main_region(occupied, resolution, radius, density):
    """Keep a coherent footprint while rejecting isolated exterior samples.

    Density is measured in a metric neighborhood. Closing connects nearby scan
    gaps before choosing the largest region; its outer contour includes interior
    samples without inventing colors in unobserved areas.
    """
    half_width = max(1, int(np.ceil(radius / resolution)))
    size = 2 * half_width + 1
    support = cv2.boxFilter(occupied.astype(np.float32), -1, (size, size),
                            borderType=cv2.BORDER_CONSTANT)
    dense = (support >= density).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if not dense.any():
        # Sparse scans can have no dense seed; use connected sample proximity.
        dense = cv2.dilate(occupied, kernel)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros_like(occupied)
    cv2.drawContours(footprint, contours, -1, 1, cv2.FILLED)
    # Recover thin boundaries near the dense seed instead of eroding the walls.
    margin = max(1, int(np.ceil(0.06 / resolution)))
    return cv2.dilate(footprint, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))


def visible_colors(rows, cols, heights, colors, width, height, resolution,
                   footprint, splat_radius):
    """Cover nearby pixels with the visible point's original RGB, without blending."""
    radius = int(np.ceil(splat_radius / resolution))
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = ((dx * resolution) ** 2 + (dy * resolution) ** 2
            <= splat_radius ** 2 + 1e-12).astype(np.uint8)
    depth = np.full(height * width, np.inf, np.float32)
    distance = np.full(height * width, np.inf, np.float32)
    rgb = np.full((height * width, 3), 245, np.uint8)
    footprint_flat = footprint.ravel()
    for oy, ox in np.argwhere(disk) - radius:
        target_rows, target_cols = rows + oy, cols + ox
        inside = ((target_rows >= 0) & (target_rows < height)
                  & (target_cols >= 0) & (target_cols < width))
        indices = np.flatnonzero(inside)
        target = target_rows[inside] * width + target_cols[inside]
        squared_distance = int(ox * ox + oy * oy)
        nearer = heights[indices] < depth[target]
        tied = ((heights[indices] == depth[target])
                & (squared_distance < distance[target]))
        take = (nearer | tied) & (footprint_flat[target] > 0)
        indices, target = indices[take], target[take]
        depth[target] = heights[indices]
        distance[target] = squared_distance
        rgb[target] = colors[indices]
    occupied = np.isfinite(depth.reshape(height, width)).astype(np.uint8)
    return rgb.reshape(height, width, 3), occupied


def fused_colors(rows, cols, heights, colors, width, height, resolution,
                 footprint, splat_radius, blend_height, confidence):
    """Gaussian splats with soft coverage edges and front-surface height gating."""
    radius = int(np.ceil(splat_radius / resolution))
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = ((dx * resolution) ** 2 + (dy * resolution) ** 2
            <= splat_radius ** 2 + 1e-12).astype(np.uint8)
    surface = np.full((height, width), np.inf, np.float32)
    surface[rows, cols] = heights
    front = cv2.erode(surface, disk).ravel()
    weights = np.zeros(height * width, np.float32)
    sums = np.zeros((height * width, 3), np.float32)
    footprint_flat = footprint.ravel()
    # The support radius spans three sigma. Subtract the edge weight so the
    # truncated kernel reaches zero smoothly instead of ending in a hard disk.
    sigma = max(splat_radius / 3, resolution * 0.5)
    edge_weight = np.exp(-0.5 * (splat_radius / sigma) ** 2)
    for oy, ox in np.argwhere(disk) - radius:
        target_rows, target_cols = rows + oy, cols + ox
        inside = ((target_rows >= 0) & (target_rows < height)
                  & (target_cols >= 0) & (target_cols < width))
        indices = np.flatnonzero(inside)
        target = target_rows[inside] * width + target_cols[inside]
        difference = heights[indices] - front[target]
        eligible = (difference <= blend_height) & (footprint_flat[target] > 0)
        indices, target = indices[eligible], target[eligible]
        difference = difference[eligible]
        spatial_weight = np.exp(-0.5 * ((ox * resolution) ** 2 + (oy * resolution) ** 2)
                                / sigma ** 2)
        if splat_radius > 0:
            spatial_weight = max(0, (spatial_weight - edge_weight) / (1 - edge_weight))
        weight = (spatial_weight
                  * np.exp(-0.5 * (difference / (blend_height / 2)) ** 2)
                  * confidence[indices])
        # Destination indices are unique within an offset, so += is sufficient.
        weights[target] += weight
        sums[target] += colors[indices] * weight[:, None]
    occupied = (weights.reshape(height, width) > 0).astype(np.uint8)
    rgb = np.full((height * width, 3), 245, np.uint8)
    valid = weights > 0
    color = sums[valid] / weights[valid, None]
    # Fade only the weakest Gaussian tails. Stronger coverage stays opaque so
    # sparse sampling does not wash otherwise observed surfaces into background.
    opacity = (np.minimum(1, weights[valid, None] / 0.01) if splat_radius > 0
               else np.ones((int(valid.sum()), 1), np.float32))
    rgb[valid] = np.clip(np.rint(color * opacity + 245 * (1 - opacity)), 0, 255).astype(np.uint8)
    return rgb.reshape(height, width, 3), occupied


def render(points, y_min, y_max, resolution, trim_percent, fill_pixels,
           cleanup=True, cleanup_radius=0.15, cleanup_density=0.1,
           splat_radius=0.02, output_size=4096, color_mode="blend", blend_height=0.04):
    valid = (np.isfinite(points).all(axis=1)
             & (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    points = points[valid]
    if not len(points):
        raise ValueError("No points remain within the requested height range")

    # Trim only the extreme horizontal tails to limit isolated scan outliers.
    low, high = np.percentile(points[:, [0, 2]],
                              [trim_percent, 100 - trim_percent], axis=0)
    if output_size:
        width = height = output_size
        padding = 24
        span = float(np.max(high - low))
        if span > 0:
            resolution = span / (output_size - 2 * padding - 1)
        center = (low + high) / 2
        origin_x = center[0] - (width - 1) / 2 * resolution
        origin_z = center[1] + (height - 1) / 2 * resolution
    else:
        padding = max(24, int(np.ceil(0.36 / resolution)))
        width, height = np.ceil((high - low) / resolution).astype(int) + 2 * padding + 1
        origin_x = low[0] - padding * resolution
        origin_z = high[1] + padding * resolution
    if width * height > 25_000_000:
        raise ValueError("Image exceeds 25 million pixels; increase --resolution")
    cols = np.rint((points[:, 0] - origin_x) / resolution).astype(int)
    rows = np.rint((origin_z - points[:, 2]) / resolution).astype(int)
    inside = ((cols >= padding) & (cols < width - padding)
              & (rows >= padding) & (rows < height - padding)
              & (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
              & (points[:, 2] >= low[1]) & (points[:, 2] <= high[1]))
    points, cols, rows = points[inside], cols[inside], rows[inside]

    pixels = rows * width + cols
    if color_mode == "blend":
        cells, inverse = np.unique(pixels, return_inverse=True)
        heights = np.full(len(cells), np.inf, np.float32)
        np.minimum.at(heights, inverse, points[:, 1])
        difference = points[:, 1] - heights[inverse]
        eligible = difference <= blend_height
        sample_weights = np.exp(-0.5 * (difference[eligible] / (blend_height / 2)) ** 2)
        cell_weights = np.bincount(inverse[eligible], weights=sample_weights,
                                   minlength=len(cells))
        colors = np.stack([
            np.bincount(inverse[eligible], weights=points[eligible, channel] * sample_weights,
                        minlength=len(cells)) / cell_weights
            for channel in (3, 4, 5)
        ], axis=1).astype(np.float32)
        confidence = np.sqrt(np.minimum(cell_weights, 4)).astype(np.float32)
    else:
        order = np.lexsort((points[:, 1], pixels))
        first = np.r_[True, np.diff(pixels[order]) != 0]
        selected = order[first]
        cells = pixels[selected]
        heights = points[selected, 1]
        colors = np.clip(points[selected, 3:6], 0, 255).astype(np.uint8)
    cell_rows, cell_cols = cells // width, cells % width
    occupied = np.zeros((height, width), np.uint8)
    occupied[cell_rows, cell_cols] = 1

    footprint = np.ones_like(occupied)
    retained = len(points)
    if cleanup:
        # Keep the density test at ~1.5 cm, independent of output resolution.
        factor = max(1, int(np.floor(0.015 / resolution + 0.5)))
        coarse = np.zeros(((height + factor - 1) // factor,
                           (width + factor - 1) // factor), np.uint8)
        coarse[cell_rows // factor, cell_cols // factor] = 1
        coarse = main_region(coarse, resolution * factor, cleanup_radius, cleanup_density)
        footprint = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)[:height, :width]
        retained = int(np.count_nonzero(footprint[rows, cols]))
    keep = footprint[cell_rows, cell_cols] > 0
    if color_mode == "blend":
        rgb, occupied = fused_colors(
            cell_rows[keep], cell_cols[keep], heights[keep], colors[keep],
            width, height, resolution, footprint, splat_radius, blend_height,
            confidence[keep])
    else:
        rgb, occupied = visible_colors(
            cell_rows[keep], cell_cols[keep], heights[keep], colors[keep],
            width, height, resolution, footprint, splat_radius)

    # Splat nearby samples into tiny holes; leave larger missing regions blank.
    if fill_pixels > 0:
        distances, labels = cv2.distanceTransformWithLabels(
            1 - occupied, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        palette = np.zeros((int(occupied.sum()) + 1, 3), np.uint8)
        covered = occupied.astype(bool)
        palette[labels[covered]] = rgb[covered]
        fill = (~covered) & (distances <= fill_pixels) & (footprint > 0)
        rgb[fill] = palette[labels[fill]]
    return rgb, retained, resolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path,
                        help="Scene metadata directory; inferred for <scene>/pointclouds/*.ply")
    cut_group = parser.add_mutually_exclusive_group()
    cut_group.add_argument("--cut-height", type=float,
                           help="Optional absolute cut in meters above the floor; overrides ratio mode")
    cut_group.add_argument("--height-ratio", type=float,
                           help="Fraction of the floor-to-ceiling height above the floor (default: 0.45)")
    parser.add_argument("--y-min", type=float,
                        help="Override the automatic cut with a world-Y minimum")
    parser.add_argument("--y-max", type=float,
                        help="Override the automatic floor limit with a world-Y maximum")
    parser.add_argument("--resolution", type=float, default=0.005,
                        help="Meters per pixel when --output-size 0 (default: 0.005)")
    parser.add_argument("--output-size", type=int, default=4096,
                        help="Square output side in pixels; 0 uses metric resolution (default: 4096)")
    parser.add_argument("--splat-radius", type=float, default=0.02,
                        help="Point coverage RADIUS in meters (default: 0.02; diameter 0.04 m)")
    parser.add_argument("--color-mode", choices=("blend", "original"), default="blend",
                        help="Height-gated RGB fusion or original visible-point colors (default: blend)")
    parser.add_argument("--blend-height", type=float, default=0.04,
                        help="Maximum height difference for same-surface fusion in meters (default: 0.04)")
    parser.add_argument("--trim-percent", type=float, default=0.1,
                        help="Horizontal percentile trimmed at each end; 0 keeps full extent")
    parser.add_argument("--fill-pixels", type=float, default=2,
                        help="Maximum distance for filling small holes; 0 disables filling")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Disable removal of points outside the main footprint")
    parser.add_argument("--cleanup-radius", type=float, default=0.15,
                        help="Neighborhood radius in meters for footprint cleanup (default: 0.15)")
    parser.add_argument("--cleanup-density", type=float, default=0.1,
                        help="Minimum local pixel occupancy for a footprint seed (default: 0.1)")
    args = parser.parse_args()
    if args.cut_height is None and args.height_ratio is None:
        args.height_ratio = 0.45
    numeric_args = [args.resolution, args.trim_percent, args.fill_pixels,
                    args.cleanup_radius, args.cleanup_density,
                    args.splat_radius, args.blend_height]
    numeric_args += [value for value in (args.y_min, args.y_max, args.cut_height,
                                         args.height_ratio) if value is not None]
    if not (np.isfinite(numeric_args).all() and args.resolution > 0
            and (args.height_ratio is None or 0 < args.height_ratio < 1)
            and (args.cut_height is None or args.cut_height > 0)
            and 0 <= args.trim_percent < 50 and args.fill_pixels >= 0
            and args.cleanup_radius > 0 and 0 < args.cleanup_density <= 1
            and args.splat_radius >= 0 and args.blend_height > 0
            and (args.output_size == 0 or args.output_size >= 64)):
        parser.error("Invalid resolution, height range, trim percentage, or fill distance")
    scene_dir = args.scene_dir
    if scene_dir is None and args.source.parent.name == "pointclouds":
        scene_dir = args.source.parent.parent
    floors, camera_y = scene_metadata(scene_dir)
    if len(floors) > 1:
        print(f"Skipping multi-floor scene: {scene_dir}; floor IDs={sorted(floors)}",
              file=sys.stderr)
        return 3
    points = read_ply(args.source)
    floor_y = ceiling_y = None
    if args.y_min is None or args.y_max is None:
        floor_y, ceiling_y = estimate_planes(points, camera_y)
        span = floor_y - ceiling_y
        cut_height = (args.cut_height if args.cut_height is not None
                      else args.height_ratio * span)
        if args.y_min is None and cut_height >= span:
            parser.error("Cut height must be below the estimated ceiling")
        auto_cut = floor_y - cut_height
        print(f"Estimated floor Y={floor_y:.4f}, ceiling Y={ceiling_y:.4f}, "
              f"height={span:.4f} m; cut={cut_height:.4f} m above floor "
              f"(Y={auto_cut:.4f})")
        if args.height_ratio is not None:
            print(f"Height ratio={args.height_ratio:g}; "
                  "cut = floor + ratio * (ceiling - floor) in upward height coordinates")
    y_min = args.y_min if args.y_min is not None else auto_cut
    y_max = args.y_max if args.y_max is not None else floor_y + 0.08
    if y_min >= y_max:
        parser.error("The retained world-Y minimum must be less than its maximum")
    print(f"Retaining world Y in [{y_min:.4f}, {y_max:.4f}]; "
          "smaller Y wins the top-down depth buffer")
    print(f"Point coverage radius={args.splat_radius:g} m "
          f"(diameter={2 * args.splat_radius:g} m); RGB mode={args.color_mode}; "
          f"same-surface fusion height tolerance={args.blend_height:g} m")
    if args.color_mode == "blend" and args.splat_radius > 0:
        print(f"Gaussian splat: nominal sigma={args.splat_radius / 3:g} m, "
              "soft coverage edges, density-dependent opacity")
    rgb, retained, effective_resolution = render(points, y_min, y_max, args.resolution,
                           args.trim_percent, args.fill_pixels,
                           cleanup=not args.no_cleanup,
                           cleanup_radius=args.cleanup_radius,
                           cleanup_density=args.cleanup_density,
                           splat_radius=args.splat_radius,
                           output_size=args.output_size,
                           color_mode=args.color_mode,
                           blend_height=args.blend_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {args.output}")
    print(f"Saved {args.output}: {rgb.shape[1]} x {rgb.shape[0]} pixels, "
          f"{effective_resolution:g} m/pixel, {retained:,}/{len(points):,} points retained")


if __name__ == "__main__":
    raise SystemExit(main())
