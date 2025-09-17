from math import sqrt

import torch


def _derive_image_information(tensor: torch.Tensor) -> tuple[int, int, int]:
    t_shp = tensor.shape
    samples_per_batch = t_shp[0]
    num_patches = t_shp[1]
    embedding_dim = t_shp[2]

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

    return samples_per_batch, patches_per_side, embedding_dim


def _reorder_patch_embeddings(embedding_tensor: torch.Tensor) -> torch.Tensor:
    image_info = _derive_image_information(embedding_tensor)
    samples_per_batch, patches_per_side, embedding_dim = image_info
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


def get_patch_embedding_without_cls_square_multilevel(
    t: list[torch.Tensor],
) -> torch.Tensor:
    """
    Reorder the output of a ViT to get each patch's embedding after every transformation step.
    This function assumes that the image was patched in an x*x raster, and that the output does not include a CLS token.
    :param t: model output: list of tensors, with each tensor having the shape (num_batches, num_patches, embedding_dim)
    :return: embeddings after each transformation step
    """
    transformation_steps = len(t)
    samples_per_batch, patches_per_side, embedding_dim = _derive_image_information(t[0])
    out_shape = (
        samples_per_batch,
        transformation_steps,
        patches_per_side,
        patches_per_side,
        embedding_dim,
    )
    tensor_stack = torch.stack(t, dim=1).reshape(out_shape)
    return tensor_stack


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
