import importlib
from unittest.mock import MagicMock, patch


def _load_backend_module():
    class DummyCDLL:
        def __getattr__(self, name):
            if name.startswith("cu"):
                return MagicMock()
            raise AttributeError(name)

    with patch("ctypes.CDLL", return_value=DummyCDLL()):
        return importlib.reload(
            importlib.import_module(
                "vllm.v1.kv_offload.fpga.backends.gpu_direct_p2p"
            )
        )


def test_resolve_map_size_uses_bar_resource_size(tmp_path):
    module = _load_backend_module()

    bar_path = tmp_path / "resource2"
    bar_path.write_bytes(b"\0" * (8 * 1024))

    resolved = module._resolve_map_size(
        str(bar_path),
        requested_map_size=0,
        default_size=1024 * 1024,
    )

    assert resolved == 8 * 1024
