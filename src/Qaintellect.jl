module Qaintellect

using Flux, Zygote, Qaintessent
using Zygote: @adjoint

include("trainable.jl")
export
    get_trainable

end
