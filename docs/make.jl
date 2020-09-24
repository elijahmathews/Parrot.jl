push!(LOAD_PATH, "../src/")

using Documenter, ProspectorML

makedocs(sitename="ProspectorML.jl")

deploydocs(
    repo = "github.com/elijahmathews/ProspectorML.jl.git",
    devbranch = "primary",
)
