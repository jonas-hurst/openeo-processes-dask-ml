from math import sqrt

import torch


def _reorder_patch_embeddings(embedding_tensor: torch.Tensor) -> torch.Tensor:
    tensor_shp = embedding_tensor.shape
    samples_per_batch = tensor_shp[0]
    num_patches = tensor_shp[1]
    try:
        patches_per_side = sqrt(num_patches)
        if patches_per_side % 1 != 0:
            raise ValueError
        patches_per_side = int(patches_per_side)
    except ValueError:
        raise Exception(
            "Postprocessing Error: Cannot arrange the model output patches into an n*n "
            "raster. If the model output includes a CLS token, use "
            "get_patch_embeddings_wit_cls_square function instead."
        )
    embedding_dim = tensor_shp[2]
    out_shape = (samples_per_batch, patches_per_side, patches_per_side, embedding_dim)
    reshaped = embedding_tensor.reshape(out_shape)
    return reshaped


def get_patch_embeddings_without_cls_square(t: list[torch.Tensor]) -> torch.Tensor:
    """
    Reorder the output of a ViT to get each patch's embedding, assuming that the image was patched in an x*x raster, and that the output does not include
    :param t: model output: list of tensors, with each tensor having the shape (num_batches, num_patches, embedding_dim)
    :return: embeddings
    """
    embedding_tensor = t.pop()
    return _reorder_patch_embeddings(embedding_tensor)


def get_patch_embeddings_with_cls_square(t: list[torch.Tensor]) -> torch.Tensor:
    """
    Reorder the output of a ViT to get each patch's embedding, assuming that the image was patched in an x*x raster, and that the output does include a CLS token
    :param t: model output: list of tensors, with each tensor having the shape (num_batches, num_patches, embedding_dim)
    :return: embeddings
    """
    embedding_tensor = t.pop()[:, 1:, :]
    return _reorder_patch_embeddings(embedding_tensor)


# def postproc_terramind_backbone_output_full(t: list[torch.Tensor]) -> torch.Tensor:
#     samples_per_batch = int(t[0].numel() / (14 * 14 * 1024))
#     num_levels = len(t)
#
#     out_shape = (samples_per_batch, num_levels, 14, 14, 1024)
#     tensor_stack = torch.stack(t, dim=1).reshape(out_shape)
#
#     return tensor_stack


def get_image_cls_embedding_prepended_torch(t: list[torch.Tensor]) -> torch.Tensor:
    """
    Returns the CLS embeddings, assuming the CLS embedding is at the first embedding position (index 0)
    :param t: ViT encoder output, a list of
    :return: The CLS embedding per batch, shape is (batch_size, embedding_size)
    """
    embedding_tensor = t.pop()
    embeddings = embedding_tensor[:, 0, :]
    return embeddings


def get_image_cls_embedding_appended_torch(t: list[torch.Tensor]) -> torch.Tensor:
    """
    Returns the CLS embeddings, assuming the CLS embedding is at the last embedding position (index -1)
    :param t: ViT encoder output, a list of
    :return: The CLS embedding per batch, shape is (batch_size, embedding_size)
    """
    embedding_tensor = t.pop()
    embeddings = embedding_tensor[:, -1, :]
    return embeddings
