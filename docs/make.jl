push!(LOAD_PATH, "../src/")

using Documenter, ProspectorML

makedocs()

deploydocs(
    repo = "github.com/elijahmathews/ProspectorML.jl.git"
)
