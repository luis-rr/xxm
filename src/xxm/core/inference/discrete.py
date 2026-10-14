from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.potentials.discrete import DiscreteChainPotentials, DiscretePotential

from .chains.discrete import DiscreteChain
from .results import InferredPosterior


def infer(
    potentials: DiscreteChainPotentials,
    local: DiscretePotential,
) -> InferredPosterior[DiscreteChainMarginals]:

    chain = DiscreteChain.from_potentials(potentials)

    chain = chain.add_local_potential(local)

    inferred = chain.forward_backward()

    return inferred
