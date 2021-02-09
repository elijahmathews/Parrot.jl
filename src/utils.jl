#
# Parrot.jl/src/utils.jl
#
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
#

"""
    splitindices(frac::Real, N::Integer)

Generate two sets of unique indices of cumulative length `N`.
Returns a tuple:

    (indices1, indices2)

where `indices1` is an array of unique indices of length `frac × N` and
`indices2` is an array of unique indices of length `(1 - frac) × N`. Can
be used for generating indices to separate training and test datasets, for
example.
"""
function splitindices(frac, N::Integer)

    @assert 0 <= frac <= 1

    # Generate unique random indices.
    base = randperm(N)

    indices1 = base[1:ceil(Int, frac * N)]
    indices2 = base[(ceil(Int, frac * N) + 1):end]

    return indices1, indices2

end

"""
    fractionalerrorquantiles(fractionalerror; quants=[0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995])

Compute the 0.05%, 0.5%, 2.5%, 50% (median), 97.5%, 99.5%, and 99.95% quantiles
for a set of spectral energy distributions (SEDs) predicted by Parrot given an
array of the fractional errors `fractionalerror`.

The selected quantiles can be changed from their defaults by passing an array to
the `quants` keyword argument.
"""
function fractionalerrorquantiles(fractionalerror; quants=[0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995])

    returnval = []

    for i in 1:length(fractionalerror)

        result = zeros(length(quants), size(fractionalerror[i], 1))

        for j in 1:size(fractionalerror[i], 1)

            for k in 1:length(quants)

                result[k,j] = quantile(fractionalerror[i][j,:], quants[k])

            end

        end

        push!(returnval, result)

    end

    return returnval

end

"""
    convert(MultivariateStats.PCA{Float32}, pca::MultivariateStats.PCA{Float64})

Convert a `Float64` PCA model `pca` to a `Float32` PCA model.
"""
function Base.convert(::Type{MultivariateStats.PCA{Float32}}, pca::MultivariateStats.PCA{Float64})
    MultivariateStats.PCA(
        convert(Array{Float32,1}, pca.mean    ),
        convert(Array{Float32,2}, pca.proj    ),
        convert(Array{Float32,1}, pca.prinvars),
        convert(Float32,          pca.tprinvar),
        convert(Float32,          pca.tvar    ),
    )
end
