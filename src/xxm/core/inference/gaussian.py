from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.potentials.gaussian import GaussianChainPotentials, GaussianPotential

from .chains.gaussian import GaussianChain
from .results import InferredPosterior


def infer(
    potentials: GaussianChainPotentials,
    local: GaussianPotential,
) -> InferredPosterior[GaussianChainMarginals]:

    chain = GaussianChain.from_potentials(potentials)

    chain = chain.add_local_potential(local)

    inferred = chain.forward_backward()

    return inferred
