import numpy as np

from robotnav.rendering.depth_render import make_uint16_depth


def test_metric_depth_uses_navdp_range_and_zero_invalid() -> None:
    depth = np.array([[0.05, 0.1, 2.3456, 5.0, 5.01, np.inf]], dtype=np.float32)
    valid = np.ones(depth.shape, dtype=bool)

    encoded, info = make_uint16_depth(depth, valid, mode="metric", metric_scale=10_000.0)

    np.testing.assert_array_equal(encoded, [[0, 1000, 23456, 50000, 0, 0]])
    assert not np.any(encoded == np.iinfo(np.uint16).max)
    assert info["depth_encoding"] == {
        "type": "uint16",
        "scale": 10_000.0,
        "valid_range": [0.1, 5.0],
        "invalid_value": 0,
    }
