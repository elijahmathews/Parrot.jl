#
# ProspectorML.jl/docs/make.jl
#
# By Elijah Mathews
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

push!(LOAD_PATH, "../src/")

using Documenter, ProspectorML

makedocs(
    sitename = "ProspectorML.jl",
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/elijahmathews/ProspectorML.jl.git",
    devbranch = "primary",
)
