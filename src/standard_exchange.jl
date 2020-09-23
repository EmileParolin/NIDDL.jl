"""Local lifting operator: takes a trace and computes a local solution."""
struct BasicLiftingOp{T} <: LocalLiftingOp where T
    # Mapping from trace vector to local problem rhs vector Σi → Ωi
    MΣitoΩi::SparseMatrixCSC{Int64,Int64}
    # Mapping from (possibly extended) solution of K to DOFs of Ω
    MKtoΩi::T
    # Local matrix (in factorized form)
    Kinv::SuiteSparse.UMFPACK.UmfpackLU{Complex{Float64},Int64}
    # Memory allocation
    fi::Vector{Complex{Float64}}
    Fi::Vector{Complex{Float64}}
    Ui::Vector{Complex{Float64}}
    ui::Vector{Complex{Float64}}
end
function BasicLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    BasicLiftingOp(MΣitoΩi, MKtoΩi, Kinv, fi, Fi, Ui, ui)
end
function BasicLiftingOp(MΣitoΩi, Kinv)
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},0)
    Ui = zeros(Complex{Float64},0)
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    BasicLiftingOp(MΣitoΩi, 1, Kinv, fi, Fi, Ui, ui)
end
function (Li::BasicLiftingOp{SparseMatrixCSC{Bool,Int64}})(Mxi)
    #Li.fi =-(Li.MΣitoΩi * Mxi)
    #Li.ui = Li.MKtoΩi * (Li.Kinv \ (transpose(Li.MKtoΩi) * Li.fi))
    mul!(Li.fi, Li.MΣitoΩi, Mxi)
    lmul!(-1, Li.fi)
    mul!(Li.Fi, transpose(Li.MKtoΩi), Li.fi)
    ldiv!(Li.Ui, Li.Kinv, Li.Fi)
    mul!(Li.ui, Li.MKtoΩi, Li.Ui)
    return Li.ui
end
function (Li::BasicLiftingOp{Int64})(Mxi)
    #Li.fi =-(Li.MΣitoΩi * Mxi)
    #Li.ui = Li.Kinv \ Li.fi
    mul!(Li.fi, Li.MΣitoΩi, Mxi)
    lmul!(-1, Li.fi)
    ldiv!(Li.ui, Li.Kinv, Li.fi)
    return Li.ui
end


"""
S is defined as

    S : x -> (γ1 - ik T0 γ0) u

where u is the solution of the local problem (+ other B.C.)

    (-Δ - k^2) u = 0        in Ω
    (γ1 + ik T0 γ0) u = x   on Σ

In the implementation, from the boundary condition on Σ in the local problem we
have

    Sx = (γ1 - ik T0 γ0) u
       = (γ1 + ik T0 γ0) u - 2ik T0 γ0 u
       = x - 2ik T0 γ0 u

Then we use the equation of the local problem, namely

    a(u,u^t) - ik t(u,u^t) = -<x,u^t>

so that the scattering operator in variational form writes

    s(x,x^t) = <x,x^t> - 2ik t(u,x^t)
             = <x,x^t> - 2 ( a(uΣ, R(x^t)) + <x,x^t> )
             =-<x,x^t> - 2 a(uΣ, R(x^t))

where R(x^t) is a Dirichlet lifting of x^t in the sub-domain.

Warning: Do we need to do something about that lifting in practice???
"""
struct ImplicitLocalScatteringOp <: LocalScatteringOp
    # Local lifting operator xi → ui = Li xi
    Li::LocalLiftingOp
    # Matrix corresponding to the equation (on Σ)
    # a(uΣ,uΣ^t) - ik t(uΣ,uΣ^t) = -<x,uΣ^t>
    KΣi::SparseMatrixCSC{Complex{Float64},Int64}
    # Memory allocation
    si::Vector{Complex{Float64}}
end
function ImplicitLocalScatteringOp(Li, KΣi)
    si = zeros(Complex{Float64}, size(KΣi,1))
    return ImplicitLocalScatteringOp(Li, KΣi, si)
end
function (Si::ImplicitLocalScatteringOp)(Mxi)
    ui = Si.Li(Mxi)
    # si = -Mxi - 2*Si.KΣi*ui
    mul!(Si.si, Si.KΣi, ui)
    lmul!(-2, Si.si)
    axpy!(-1, Mxi, Si.si)
    return Si.si
end

struct ExplicitLocalScatteringOp <: LocalScatteringOp
    # Matrix of DtN operator
    Λi::Matrix{Complex{Float64}}
    # Matrix of transmission operator
    Ti::Matrix{Complex{Float64}}
    # Local lifting operator xi → ui = Li xi
    Li::LocalLiftingOp
    # Matrix of local scattering operator
    Si::Matrix{Complex{Float64}}
    # Memory allocation
    si::Vector{Complex{Float64}}
end
function ExplicitLocalScatteringOp(Λi, Ti, Li)
    Si = (Λi - conj.(Ti)) * inv(Λi + Ti)
    si = zeros(Complex{Float64}, size(Si,1))
    return ExplicitLocalScatteringOp(Λi, Ti, Li, Si, si)
end
function (Si::ExplicitLocalScatteringOp)(Mxi)
    Si.Li(Mxi) # To store local solution
    mul!(Si.si, Si.Si, Mxi)
end


"""
This local exchange op takes the form of (for 2 domains, no cross-points)

     0  -M1
    -M2  0
"""
struct BasicExchangeOp <: LocalExchangeOp
    # Local part of exchange matrix
    Πi::SparseMatrixCSC{Complex{Float64},Int64}
    # Memory allocation
    Πxi::Vector{Complex{Float64}}
end
function BasicExchangeOp(m, fullpb, pbs, i, RΣ, RΣi, MΣitoΣmt, Cwsst)
    # Local part of projection operator
    Pi = hcat([RΣi * transpose(RΣ) * Cwsst * RΣ * transpose(restriction(m, transmission_boundary(pb), dofdim(pb)))
               for pb in pbs]...)
    # Local part of symmetry operator
    Πi = transpose(MΣitoΣmt) - 2 * Pi
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return BasicExchangeOp(Πi, Πxi)
end
(Πi::BasicExchangeOp)(x) = mul!(Πi.Πxi, Πi.Πi, x)
