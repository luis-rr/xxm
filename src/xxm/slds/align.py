"""Align fitted SLDS models to synthetic ground truth."""

import jax

from xxm.core.align import align_affine, match_states_to_true
from xxm.core.data import Dataset, DatasetCollection
from xxm.slds.model import GaussianSLDS


def align_to_truth(
    model: GaussianSLDS,
    data: Dataset,
    true_states: jax.Array,
    true_latents: jax.Array,
    *,
    num_inference_iters: int = 50,
) -> GaussianSLDS:
    """Align fitted SLDS labels and latent coordinates to synthetic truth."""

    inference = model.infer(
        DatasetCollection.from_datasets((data,)),
        num_iters=num_inference_iters,
    )

    permutation = match_states_to_true(
        inference.posterior.discrete.state_probs.reshape((-1, model.num_states)),
        true_states.reshape(-1),
    )
    model = model.permute_states(permutation)

    inference = model.infer(
        DatasetCollection.from_datasets((data,)),
        num_iters=num_inference_iters,
    )

    alignment = align_affine(
        source=inference.posterior.continuous.means.reshape((-1, model.latent_dim)),
        target=true_latents.reshape((-1, model.latent_dim)),
    )

    return model.align(alignment)
