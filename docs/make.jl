# 
# Parrot.jl/docs/make.jl
# 
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
# 

push!(LOAD_PATH, "../src/")

using Documenter, Parrot

makedocs(
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
    ),
    sitename = "Parrot",
    pages = [
        "Home" => "index.md"
    ],
)

deploydocs(
    repo = "github.com/elijahmathews/Parrot.jl.git",
    devbranch = "primary",
)

