"""Align fitted LDS results to known latents."""

import jax

from xxm.core.align import align_affine
from xxm.lds.model import GaussianLDS, Inferred


def align_to_latents(
    inferred: Inferred[GaussianLDS],
    target_latents: jax.Array,
) -> Inferred[GaussianLDS]:
    source_latents = inferred.latent_posterior().mean.get()

    alignment = align_affine(
        source=source_latents,
        target=target_latents,
    )

    return inferred.align(alignment)
