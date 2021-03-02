using Documenter, Qaintellect, Qaintessent


makedocs(
    sitename="Qaintellect.jl Documentation",
    pages = [
        "Home" => "index.md",
        "Section" => [
            "flux_integration.md",
            "qaoa.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/Qaintum/Qaintellect.jl.git",
    push_preview = true
)
