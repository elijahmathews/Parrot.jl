# 
# Parrot.jl/src/utils.jl
# 
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
# 

"""
    trainvalidindices(validfrac, N::Integer)

Generate sets of training and validation indices for datasets of length `N`.
Returns a tuple:

    (validindex, trainindex)

where `validindex` is an array of unique indices of length `validfrac × N` and
`trainindex` is an array of unique indices of length `(1 - validfrac) × N`.
"""
function trainvalidindices(validfrac, N::Integer)
    
    # Generate unique random indices.
    base = randperm(N)
    
    validindex = base[1:ceil(Int, validfrac * N)]
    trainindex = base[(ceil(Int,validfrac * N) + 1):end]
    
    return validindex, trainindex
    
end

"""
    fractionalerrorquantiles(...)

"""
function fractionalerrorquantiles(fractionalerror; quants=[0.0005, 0.005, 0.025, 0.5, 0.975, 0.99, 0.999])
    
    returnval = []
    
    for i in 1:length(fractionalerror)
        
        result = zeros(length(quants), size(fractionalerror[i], 2))
        
        for j in 1:size(fractionalerror[i], 2)
            
            for k in 1:length(quants)
                
                result[k,j] = quantile(fractionalerror[i][:,j], quants[k])
                
            end
            
        end
        
        push!(returnval, result)
        
    end
    
    return returnval
    
end

function convert(MultivariateStats.PCA{Float32}, pca::MultivariateStats.PCA{Float64})
    MultivariateStats.PCA(
        convert(Array{Float32,1}, pca.mean    ),
        convert(Array{Float32,2}, pca.proj    ),
        convert(Array{Float32,1}, pca.prinvars),
        convert(Float32,          pca.tprinvar),
        convert(Float32,          pca.tvar    ),
    )
end
