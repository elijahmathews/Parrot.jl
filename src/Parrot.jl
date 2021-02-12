#
# Parrot.jl/src/Parrot.jl
#
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
#

module Parrot

    #
    # State packages we're using.
    #
    using Flux, MultivariateStats, OnlineStats, Random, Statistics

    #
    # Export really useful things from Parrot.
    #
    export Alsing, TransformPCA, ReconstructPCA, Normalize, Denormalize, convert

    #
    # Include source code files.
    #
    include("layers.jl")
    # include("prospect.jl")
    include("utils.jl")

end

