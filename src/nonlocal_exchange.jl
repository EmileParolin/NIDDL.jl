"""
Local lifting operator: takes a trace and computes a local solution.

It is parametrized with T for efficiency reasons.
"""
struct XPLiftingOp{T} <: LocalLiftingOp where T
    # Mapping from trace vector to local problem rhs vector Σi → Ωi
    MΣitoΩi::SparseMatrixCSC{Int64,Int64}
    # Mapping from (possibly extended) solution of K to DOFs of Ω
    MKtoΩi::T
    # Local matrix (in factorized form)
    Kinv::SuiteSparse.UMFPACK.UmfpackLU{Complex{Float64},Int64}
    # Transmission operator
    Ti::SparseMatrixCSC{Complex{Float64},Int64}
    # Memory allocation
    xi::Vector{Complex{Float64}}
    fi::Vector{Complex{Float64}}
    Fi::Vector{Complex{Float64}}
    Ui::Vector{Complex{Float64}}
    ui::Vector{Complex{Float64}}
end
function XPLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti)
    xi = zeros(Complex{Float64},size(MΣitoΩi,2))
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    XPLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti, xi, fi, Fi, Ui, ui)
end
function XPLiftingOp(MΣitoΩi, Kinv, Ti)
    xi = zeros(Complex{Float64},size(MΣitoΩi,2))
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},0)
    Ui = zeros(Complex{Float64},0)
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    XPLiftingOp(MΣitoΩi, 1, Kinv, Ti, xi, fi, Fi, Ui, ui)
end
function (Li::XPLiftingOp{SparseMatrixCSC{Bool,Int64}})(xi)
    # Li.fi =-(Li.MΣitoΩi * (Li.Ti * xi))
    # Li.ui = Li.MKtoΩi * (Li.Kinv \ (transpose(Li.MKtoΩi) * Li.fi))
    mul!(Li.xi, Li.Ti, xi)
    mul!(Li.fi, Li.MΣitoΩi, Li.xi)
    lmul!(-1, Li.fi)
    mul!(Li.Fi, transpose(Li.MKtoΩi), Li.fi)
    ldiv!(Li.Ui, Li.Kinv, Li.Fi)
    mul!(Li.ui, Li.MKtoΩi, Li.Ui)
    return Li.ui
end
function (Li::XPLiftingOp{Int64})(xi)
    # Li.fi =-(Li.MΣitoΩi * (Li.Ti * xi))
    # Li.ui = Li.Kinv \ Li.fi
    mul!(Li.xi, Li.Ti, xi)
    mul!(Li.fi, Li.MΣitoΩi, Li.xi)
    lmul!(-1, Li.fi)
    ldiv!(Li.ui, Li.Kinv, Li.fi)
    return Li.ui
end

"""
Local scattering operator.
"""
struct XPScatteringOp <: LocalScatteringOp
    # Local lifting operator xi → ui = Li xi
    Li::XPLiftingOp
    # Memory allocation
    si::Vector{Complex{Float64}}
end
function XPScatteringOp(Li)
    si = zeros(Complex{Float64}, size(Li.MΣitoΩi,2))
    return XPScatteringOp(Li, si)
end
function (Si::XPScatteringOp)(xi)
    ui = Si.Li(xi)
    # si = -xi + 2im * transpose(Si.Li.MΣitoΩi) * ui
    mul!(Si.si, transpose(Si.Li.MΣitoΩi), ui)
    lmul!(2im, Si.si)
    axpy!(-1, xi, Si.si)
    return Si.si
end


mutable struct ImplicitXPExchangeOp <: LocalExchangeOp
    # Mapping Σi to Σ multi-trace
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Mapping Σi to Σ single-trace
    MΣitoΣst::SparseMatrixCSC{Bool,Int64}
    # Transmission operator
    Ti::Matrix{Complex{Float64}}
    # Local part of projection operator
    invTi::Matrix{Complex{Float64}}
    # Dummy initialisation of global projection operator and preconditionner
    PA::LinearMap{Float64}
    PP::LinearMap{Float64}
    # Log for min, max, sum (to compute mean) inner cg iterations
    cg_min::Int64
    cg_max::Int64
    cg_sum::Int64
    # Memory allocation
    xst::Vector{Complex{Float64}}
    bst::Vector{Complex{Float64}}
    tmpst::Vector{Complex{Float64}}
    pxi::Vector{Complex{Float64}} # Solution local projection
    Πxi::Vector{Complex{Float64}} # Solution local exchange
    ai::Vector{Complex{Float64}} # Local allocation matvec and precond
    bi::Vector{Complex{Float64}} # Local allocation matvec and precond
end
function ImplicitXPExchangeOp(m, fullpb, pbs, i, MΣitoΣst, MΣitoΣmt, Ti)
    # Factorisation of Ti
    invTi = inv(Matrix(Ti))
    # Dummy initialisation of global projection operator and preconditionner
    PA = LinearMap{Float64}(x->x, 0)
    PP = LinearMap{Float64}(x->x, 0)
    # Memory allocation
    NΣst = size(MΣitoΣst,1)
    NΣi = size(MΣitoΣst,2)
    xst = zeros(Complex{Float64}, i==1 ? NΣst : 0)
    bst = zeros(Complex{Float64}, i==1 ? NΣst : 0)
    tmpst = zeros(Complex{Float64}, i==1 ? NΣst : 0)
    pxi = zeros(Complex{Float64}, NΣi)
    Πxi = zeros(Complex{Float64}, NΣi)
    ai = zeros(Complex{Float64}, NΣi)
    bi = zeros(Complex{Float64}, NΣi)
    return ImplicitXPExchangeOp(MΣitoΣmt, MΣitoΣst, Ti, invTi, PA, PP, 2^63-1, 0, 0,
                                xst, bst, tmpst, pxi, Πxi, ai, bi)
end
function (Πi::ImplicitXPExchangeOp)(x)
    # 2 .* Πi.pxi .- Πi.MΣitoΣmt * x
    mul!(Πi.Πxi, transpose(Πi.MΣitoΣmt), x)
    lmul!(-1, Πi.Πxi)
    axpy!(2, Πi.pxi, Πi.Πxi)
    return Πi.Πxi
end

mutable struct ExplicitXPExchangeOp <: LocalExchangeOp
    # Mapping Σi to Σ multi-trace
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Mapping Σi to Σ single-trace
    MΣitoΣst::SparseMatrixCSC{Bool,Int64}
    # Transmission operator
    Ti::Matrix{Complex{Float64}}
    # Local part of projection operator
    Πi::Matrix{Complex{Float64}}
    # Memory allocation
    Πxi::Vector{Complex{Float64}}
end
function ExplicitXPExchangeOp(m, fullpb, pbs, i, MΣitoΣst, MΣitoΣmt, Ti)
    # Memory allocation of projection operator
    Πi = zeros(Complex{Float64},size(transpose(MΣitoΣmt)))
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return ExplicitXPExchangeOp(MΣitoΣmt, MΣitoΣst, Ti, Πi, Πxi)
end
(Πi::ExplicitXPExchangeOp)(x) = mul!(Πi.Πxi, Πi.Πi, x)


struct STProjector{T<:AbstractLocalData}
    lds::Array{T,1}
    # Memory allocation
    tmpst::Vector{Complex{Float64}}
end
function STProjector(lds::Array{T,1}, Nst::Int64) where T<:AbstractLocalData
    # Memory allocation
    tmpst = zeros(Complex{Float64}, Nst)
    return STProjector(lds, tmpst)
end
function (P::STProjector{T})(y, x) where T<:AbstractLocalData
    fill!(y, 0)
    for ld in P.lds # TODO parallelize
        Πi = ld.Πi
        #y += Πi.MΣitoΣst * Πi.Ti * transpose(Πi.MΣitoΣst) * x
        mul!(Πi.ai, transpose(Πi.MΣitoΣst), x)
        mul!(Πi.bi, Πi.Ti, Πi.ai)
        mul!(P.tmpst, Πi.MΣitoΣst, Πi.bi)
        axpy!(1, P.tmpst, y)
    end
    return y
end

struct STProjector_Precond{T<:AbstractLocalData}
    lds::Array{T,1}
    # Diagonal matrix of junction weights at transmission DOFs
    Cws::SparseMatrixCSC{Float64,Int64}
    # Memory allocation
    yst::Vector{Complex{Float64}}
    cxst::Vector{Complex{Float64}}
    tmpst::Vector{Complex{Float64}}
end
function STProjector_Precond(lds::Array{T,1}, Nst::Int64, Cws) where T<:AbstractLocalData
    # Weights: need to add physcial boundary weight (=1)
    Cws_corr = sparse(UniformScaling{Float64}(1), size(Cws)...)
    icw,jcw,vcw = findnz(Cws)
    Cws_corr[(jcw.-1).*size(Cws,1).+icw] .= vcw
    # Memory allocation
    yst = zeros(Complex{Float64}, Nst)
    cxst = zeros(Complex{Float64}, Nst)
    tmpst = zeros(Complex{Float64}, Nst)
    return STProjector_Precond(lds, Cws_corr, yst, cxst, tmpst)
end
function ldiv!(y, P::STProjector_Precond{T}, x) where T<:AbstractLocalData
    fill!(P.yst, 0)
    mul!(P.cxst, P.Cws, x)
    for ld in P.lds # TODO parallelize
        Πi = ld.Πi
        #y += P.Cws * Πi.MΣitoΣst * (Πi.invTi * (transpose(Πi.MΣitoΣst) * P.Cws * x))
        mul!(Πi.ai, transpose(Πi.MΣitoΣst), P.cxst)
        mul!(Πi.bi, Πi.invTi, Πi.ai)
        mul!(P.tmpst, Πi.MΣitoΣst, Πi.bi)
        axpy!(1, P.tmpst, P.yst)
    end
    mul!(y, P.Cws, P.yst)
    return y
end


"""
Solving projection problem by using preconditionned CG method.
"""
function setup_exchange(lds::Vector{LocalData{TS,
                                              TΠ}}, x) where TS<:LocalScatteringOp where TΠ<:ImplicitXPExchangeOp
    Π1 = lds[1].Πi
    # Computation of RHS
    lmul!(0, Π1.bst)
    for ld in lds # TODO parallelize
        Πi = ld.Πi
        mul!(Πi.ai, transpose(Πi.MΣitoΣmt), x)
        mul!(Πi.bi, Πi.Ti, Πi.ai)
        mul!(Π1.tmpst, Πi.MΣitoΣst, Πi.bi)
        axpy!(1, Π1.tmpst, Π1.bst)
    end
    # Solve problem
    lmul!(0, Π1.xst)
    if length(Π1.PP) == 0 # No preconditioning
        Π1.xst, ch = cg!(Π1.xst, Π1.PA, Π1.bst;
                         tol=1.e-14, verbose=true, log=true)
    else
        Π1.xst, ch = cg!(Π1.xst, Π1.PA, Π1.bst; Pl= Π1.PP.f,
                         tol=1.e-14, verbose=true, log=true)
    end
    # Logging
    if ch.iters > 0 Π1.cg_min = min(Π1.cg_min, ch.iters) end
    Π1.cg_max = max(Π1.cg_max, ch.iters)
    Π1.cg_sum += ch.iters
    # Local storage
    for ld in lds
        Πi = ld.Πi
        mul!(Πi.pxi, transpose(Πi.MΣitoΣst), Π1.xst)
    end
end
