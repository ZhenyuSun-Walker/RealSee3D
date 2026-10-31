"""Typed, chunked PLY I/O for Realsee3D vertex-only point clouds.

Binary files use packed little-endian records. The standard XYZ/RGB/observer
schema occupies exactly 17 bytes per vertex, without native struct padding.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import tempfile

import numpy as np


TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
STANDARD_PROPERTIES = (
    ("float", "x"), ("float", "y"), ("float", "z"),
    ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
    ("ushort", "observer_index"),
)


@dataclass(frozen=True)
class Header:
    format: str
    count: int
    properties: tuple
    offset: int
    comments: tuple = ()

    @property
    def dtype(self):
        endian = ">" if self.format == "binary_big_endian" else "<"
        return np.dtype([(name, endian + TYPES[kind]) for kind, name in self.properties])


def read_header(stream):
    if stream.readline().strip() != b"ply":
        raise ValueError("Not a PLY file")
    encoding = None
    count = None
    properties = []
    comments = []
    vertex_element = False
    for _ in range(10000):
        line = stream.readline()
        if not line:
            raise ValueError("Truncated PLY header")
        fields = line.decode("ascii").strip().split()
        if not fields:
            continue
        if fields[0] == "format":
            if len(fields) != 3 or fields[2] != "1.0":
                raise ValueError("Unsupported PLY format version")
            encoding = fields[1]
        elif fields[0] == "element":
            if len(fields) != 3 or int(fields[2]) < 0:
                raise ValueError("Invalid PLY element")
            vertex_element = fields[1] == "vertex"
            if vertex_element:
                if count is not None:
                    raise ValueError("Duplicate vertex element")
                count = int(fields[2])
            elif int(fields[2]) != 0:
                raise ValueError("Only vertex-only PLY point clouds are supported")
        elif fields[0] == "property" and vertex_element:
            if len(fields) != 3 or fields[1] not in TYPES:
                raise ValueError("Unsupported vertex property (list properties are not supported)")
            if any(name == fields[2] for _, name in properties):
                raise ValueError("Duplicate PLY property")
            properties.append((fields[1], fields[2]))
        elif fields[0] in ("comment", "obj_info"):
            comments.append(line.decode("ascii").rstrip("\r\n"))
        elif fields[0] == "end_header":
            if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
                raise ValueError("Unsupported or missing PLY format")
            if count is None or not properties:
                raise ValueError("Missing vertex schema")
            return Header(encoding, count, tuple(properties), stream.tell(), tuple(comments))
    raise ValueError("PLY header exceeds 10000 lines")


def header_bytes(count, properties=STANDARD_PROPERTIES, comments=()):
    lines = ["ply", "format binary_little_endian 1.0", *comments,
             f"element vertex {count}"]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    return "\n".join([*lines, "end_header", ""]).encode("ascii")


def iter_chunks(stream, header, chunk_vertices=250000):
    """Validate every declared row/byte and yield records at their declared types."""
    if chunk_vertices < 1:
        raise ValueError("Chunk size must be positive")
    remaining = header.count
    while remaining:
        count = min(remaining, chunk_vertices)
        if header.format == "ascii":
            values = np.loadtxt(stream, dtype=np.float64, max_rows=count,
                                ndmin=2, comments=None)
            if values.shape != (count, len(header.properties)):
                raise ValueError("ASCII PLY row count/schema does not match its header")
            records = np.empty(count, dtype=header.dtype)
            for index, (_, name) in enumerate(header.properties):
                column = values[:, index]
                dtype = records.dtype[name]
                if dtype.kind in "iu":
                    limits = np.iinfo(dtype)
                    if (not np.isfinite(column).all() or np.any(column < limits.min)
                            or np.any(column > limits.max) or np.any(column != np.trunc(column))):
                        raise ValueError(f"Invalid integer values for property {name}")
                records[name] = column
        else:
            records = np.fromfile(stream, dtype=header.dtype, count=count)
            if len(records) != count:
                raise ValueError("Truncated binary PLY payload")
        remaining -= count
        yield records
    trailer = stream.read()
    if trailer and (header.format != "ascii" or trailer.strip()):
        raise ValueError("Unexpected data after declared PLY vertices")


def validate_binary(path):
    """Check exact binary payload length, count, field order, and byte widths."""
    path = Path(path)
    with path.open("rb") as stream:
        header = read_header(stream)
        if header.format != "binary_little_endian":
            raise ValueError("Expected binary_little_endian PLY")
        expected = header.offset + header.count * header.dtype.itemsize
        if os.fstat(stream.fileno()).st_size != expected:
            raise ValueError("Binary PLY size does not match its declared vertex layout")
    return header


@contextmanager
def path_lock(path):
    path = Path(path)
    with (path.parent / ("." + path.name + ".lock")).open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _replace(temp, target):
    os.replace(temp, target)
    descriptor = os.open(str(Path(target).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_binary_ply(path, points, colors, observer_indices, chunk_vertices=250000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(points)
    if points.shape != (count, 3) or colors.shape != (count, 3) or len(observer_indices) != count:
        raise ValueError("Point, color, and observer dimensions do not match")
    dtype = Header("binary_little_endian", count, STANDARD_PROPERTIES, 0).dtype
    with path_lock(path):
        descriptor, temp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(header_bytes(count))
                for start in range(0, count, chunk_vertices):
                    stop = min(count, start + chunk_vertices)
                    rgb = np.asarray(colors[start:stop])
                    observers = np.asarray(observer_indices[start:stop])
                    if (not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255)
                            or np.any(rgb != np.trunc(rgb)) or not np.isfinite(observers).all()
                            or np.any(observers < 0) or np.any(observers > 65535)
                            or np.any(observers != np.trunc(observers))):
                        raise ValueError("RGB or observer_index is outside its declared integer type")
                    records = np.empty(stop - start, dtype=dtype)
                    for axis, name in enumerate(("x", "y", "z")):
                        records[name] = points[start:stop, axis]
                    for axis, name in enumerate(("red", "green", "blue")):
                        records[name] = rgb[:, axis]
                    records["observer_index"] = observers
                    records.tofile(stream)
                stream.flush()
                os.fsync(stream.fileno())
            validate_binary(temp)
            os.chmod(temp, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o640)
            _replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def convert_to_binary(path, chunk_vertices=250000):
    """Atomically convert in place, verifying SHA-256 before replacing the source."""
    path = Path(path)
    with path_lock(path):
        original = path.stat()
        with path.open("rb") as source:
            header = read_header(source)
            if header.format == "binary_little_endian":
                validate_binary(path)
                return {"path": str(path), "action": "already_binary", "vertices": header.count,
                        "before_bytes": original.st_size, "after_bytes": original.st_size}
            dtype = header.dtype.newbyteorder("<")
            descriptor, temp = tempfile.mkstemp(prefix="." + path.name + ".binary.", suffix=".tmp", dir=path.parent)
            try:
                expected_hash = hashlib.sha256()
                with os.fdopen(descriptor, "wb") as target:
                    target.write(header_bytes(header.count, header.properties, header.comments))
                    for records in iter_chunks(source, header, chunk_vertices):
                        packed = records.astype(dtype, copy=False).tobytes()
                        expected_hash.update(packed)
                        target.write(packed)
                    target.flush()
                    os.fsync(target.fileno())
                converted = validate_binary(temp)
                if converted.count != header.count or converted.properties != header.properties:
                    raise ValueError("Converted PLY schema/count changed")
                actual_hash = hashlib.sha256()
                with open(temp, "rb") as check:
                    check.seek(converted.offset)
                    while True:
                        block = check.read(8 * 1024 * 1024)
                        if not block:
                            break
                        actual_hash.update(block)
                if actual_hash.digest() != expected_hash.digest():
                    raise ValueError("Converted PLY payload failed SHA-256 verification")
                current = path.stat()
                if ((current.st_ino, current.st_size, current.st_mtime_ns)
                        != (original.st_ino, original.st_size, original.st_mtime_ns)):
                    raise ValueError("Source PLY changed during conversion")
                os.chmod(temp, stat.S_IMODE(original.st_mode))
                result = {"path": str(path), "action": "converted", "vertices": header.count,
                          "before_bytes": original.st_size, "after_bytes": os.path.getsize(temp),
                          "payload_sha256": actual_hash.hexdigest()}
                _replace(temp, path)
                return result
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)


def read_xyzrgb(path, chunk_vertices=250000):
    """Read ASCII or either binary byte order through a single open descriptor."""
    names = ("x", "y", "z", "red", "green", "blue")
    with Path(path).open("rb") as stream:
        header = read_header(stream)
        if not all(name in header.dtype.names for name in names):
            raise ValueError("PLY must contain XYZ and RGB properties")
        if header.count == 0:
            raise ValueError("Point cloud is empty")
        points = np.empty((header.count, 6), np.float32)
        offset = 0
        for records in iter_chunks(stream, header, chunk_vertices):
            for column, name in enumerate(names):
                points[offset:offset + len(records), column] = records[name]
            offset += len(records)
        return points
