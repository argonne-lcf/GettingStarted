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
import ctypes
import sys

_SIZE_T = struct.Struct("=Q")  # native-endian unsigned 8-byte (matches size_t)

class StringKeyPickler:
    """Pickler matching dragon::SerializableString serialization format."""

    def dumps(self, key) -> bytes:
        body = str(key).encode("utf-8")
        return _SIZE_T.pack(len(body)) + body

    def loads(self, blob) -> str:
        n = _SIZE_T.unpack_from(blob, 0)[0]
        return bytes(blob[_SIZE_T.size : _SIZE_T.size + n]).decode("utf-8")

class NumPy2DPickler:

    def __init__(self, data_type: np.dtype):
        self._data_type = data_type

    def dump(self, nparr, file) -> None:

        # write the dimension of the array
        size_t_size = ctypes.sizeof(ctypes.c_size_t)
        nrow, ncol = nparr.shape
        bytes_nrow = nrow.to_bytes(size_t_size, byteorder=sys.byteorder)
        bytes_ncol = ncol.to_bytes(size_t_size, byteorder=sys.byteorder)
        file.write(bytes_nrow)

        # Write the 2D array as a sequence of 1D array rows. Numpy's
        # default is that rows are guaranteed contiguous. If your
        # application had completely contiguous data, then crafting
        # a new pickler and writing your own C++ serializable class
        # would be in order. Otherwise, this should work for most
        # default numpy matrices.

        for i in range(nrow):
            mv = memoryview(nparr[i])
            bobj = mv.tobytes()
            file.write(bytes_ncol) # The C++ (de)serialization for vector 1D expects this.
            file.write(bobj)


    def load(self, file):

        obj = None

        # read the dimension of the array
        size_t_size = ctypes.sizeof(ctypes.c_size_t)
        nrow = int.from_bytes(file.read(size_t_size), sys.byteorder)
        item_size = np.dtype(self._data_type).itemsize

        try:
            while True:
                ncol = int.from_bytes(file.read(size_t_size), sys.byteorder)
                data = file.read(ncol*item_size)
                if obj is None:
                    # convert bytes to bytearray
                    view = memoryview(data)
                    obj = bytearray(view)
                else:
                    obj.extend(data)
        except EOFError:
            pass

        ret_arr = np.frombuffer(obj, dtype=self._data_type).reshape((nrow, ncol))

        return ret_arr


class NumPy1DPickler:

    def __init__(self, data_type: np.dtype):
        self._data_type = data_type

    def dump(self, nparr, file) -> None:

        # write the dimension of the array
        size_t_size = ctypes.sizeof(ctypes.c_size_t)
        ncol = nparr.shape[0]
        bytes_ncol = ncol.to_bytes(size_t_size, byteorder=sys.byteorder)
        file.write(bytes_ncol)

        mv = memoryview(nparr[0])
        bobj = mv.tobytes()
        file.write(bytes_ncol) # The C++ (de)serialization for vector 1D expects this.
        file.write(bobj)


    def load(self, file):

        obj = None

        # read the dimension of the array
        size_t_size = ctypes.sizeof(ctypes.c_size_t)
        ncol = int.from_bytes(file.read(size_t_size), sys.byteorder)
        item_size = np.dtype(self._data_type).itemsize

        try:
            data = file.read(ncol*item_size)
            if obj is None:
                # convert bytes to bytearray
                view = memoryview(data)
                obj = bytearray(view)
            else:
                obj.extend(data)
        except EOFError:
            pass

        ret_arr = np.frombuffer(obj, dtype=self._data_type)

        return ret_arr
