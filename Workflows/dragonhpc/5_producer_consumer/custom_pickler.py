# ---------------------------------------------------------------------------
# Custom picklers matching the C++ client's serialization format
#
# The C++ producer writes keys as `dragon::SerializableString` and values as
# `dragon::SerializableDouble2DVector`. Those classes use their own binary
# layout (not Python pickle), so cloudpickle.loads chokes on the bytes. We
# replicate that layout here with `dumps`/`loads` (key) and `dump`/`load`
# (value) so the Python client can interoperate with the C++ client.
#
# Serialization formats (from dragon/src/lib/serializable.cpp):
#   SerializableString       : size_t length || raw bytes  (native endian)
#   SerializableDouble2DVector: size_t nrows || size_t ncols || nrows*ncols
#                               float64 values, row-major (native endian)
# ---------------------------------------------------------------------------

import struct
import numpy as np

_SIZE_T = struct.Struct("=Q")  # native-endian unsigned 8-byte (matches size_t)

class StringKeyPickler:
    """Pickler matching dragon::SerializableString serialization format."""

    def dumps(self, key) -> bytes:
        body = str(key).encode("utf-8")
        return _SIZE_T.pack(len(body)) + body

    def loads(self, blob) -> str:
        n = _SIZE_T.unpack_from(blob, 0)[0]
        return bytes(blob[_SIZE_T.size : _SIZE_T.size + n]).decode("utf-8")


def _read_exact(file, n: int) -> bytes:
    """Pull exactly ``n`` bytes off the FLI stream, looping over partial reads."""
    buf = bytearray()
    while len(buf) < n:
        chunk = file.read(n - len(buf))
        if not chunk:
            raise EOFError(f"expected {n} bytes, got {len(buf)}")
        buf.extend(chunk)
    return bytes(buf)


class Double2DValuePickler:
    """Pickler matching dragon::SerializableDouble2DVector serialization format."""

    def dump(self, arr, file) -> None:
        a = np.ascontiguousarray(arr, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(
                f"Double2DValuePickler requires a 2D array; got shape {a.shape}"
            )
        nrows, ncols = a.shape
        file.write(_SIZE_T.pack(nrows))
        file.write(_SIZE_T.pack(ncols))
        file.write(a.tobytes(order="C"))

    def load(self, file):
        nrows = _SIZE_T.unpack(_read_exact(file, _SIZE_T.size))[0]
        ncols = _SIZE_T.unpack(_read_exact(file, _SIZE_T.size))[0]
        if nrows == 0 or ncols == 0:
            return np.empty((nrows, ncols), dtype=np.float64)
        body = _read_exact(file, nrows * ncols * 8)
        return np.frombuffer(body, dtype=np.float64).reshape(nrows, ncols)