import numpy as np

def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=[1.6, 3.9, 1.56],
                            rotations=[0, np.pi / 2],
                            dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz
    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    Source:
        https://github.com/traveller59/second.pytorch
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    res = np.transpose(ret, [2, 1, 0, 3, 4, 5])
    return res