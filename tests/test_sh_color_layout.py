import numpy as np

from robotnav.rendering.render_one_view import decode_ply_sh_coefficients


def test_decode_ply_sh_coefficients_transposes_channel_major_rest() -> None:
    f_dc = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    # PLY storage is all R, then all G, then all B coefficients.
    f_rest = np.array([[101.0, 102.0, 103.0, 201.0, 202.0, 203.0, 301.0, 302.0, 303.0]])

    decoded = decode_ply_sh_coefficients(f_dc, f_rest, sh_degree=1)

    np.testing.assert_array_equal(
        decoded,
        np.array(
            [
                [
                    [10.0, 20.0, 30.0],
                    [101.0, 201.0, 301.0],
                    [102.0, 202.0, 302.0],
                    [103.0, 203.0, 303.0],
                ]
            ],
            dtype=np.float32,
        ),
    )
