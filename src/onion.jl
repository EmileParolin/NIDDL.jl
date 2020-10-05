"""
Contains DDM parameters, based on point-to-point standard exchange.
"""
struct OnionDDM <: DDM_Type
    implicit::Bool
    OnionDDM(;implicit=true) = new(implicit)
end


struct OnionGlobalData <: AbstractGlobalData end
OnionGlobalData(args...; kwargs...) = OnionGlobalData()


struct OnionLocalData <: AbstractLocalData
    # Local to global solution mappings Ωi → Ω
    MΩitoΩ::Mapping
    # Mapping from local trace vector to global trace vector Σi → ΣMT
    MΣitoΣmt::Mapping
    # Correction matrix used when computing lifting
    corr::Vector{Float64}
    Li::LocalLiftingOp
    Si::LocalScatteringOp
    Πi::LocalExchangeOp
    Fi::Array{Complex{Float64},1}
    bi::Array{Complex{Float64},1}
    # Timer
    to::TimerOutput
end


"""Local lifting operator: takes a trace and computes a local solution."""
struct OnionLocalLiftingOp <: LocalLiftingOp
    # Mapping from trace vector to local problem rhs vector Σi → Ωi
    MΣitoΩi::Mapping
    # Mapping from (possibly extended) solution of K to DOFs of Ω
    MKtoΩi::Mapping
    # Local matrix (in factorized form)
    Kinv::SuiteSparse.UMFPACK.UmfpackLU{Complex{Float64},Int64}
    # Memory allocation
    fi::Vector{Complex{Float64}}
    Fi::Vector{Complex{Float64}}
    Ui::Vector{Complex{Float64}}
    ui::Vector{Complex{Float64}}
end
function OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, fi, Fi, Ui, ui)
end
function (Li::OnionLocalLiftingOp)(Mxi)
    #Li.fi =-(Li.MΣitoΩi * Mxi)
    #Li.ui = Li.MKtoΩi * (Li.Kinv \ (transpose(Li.MKtoΩi) * Li.fi))
    mul!(Li.fi, Li.MΣitoΩi, Mxi)
    lmul!(-1, Li.fi)
    mul!(Li.Fi, transpose(Li.MKtoΩi), Li.fi)
    ldiv!(Li.Ui, Li.Kinv, Li.Fi)
    mul!(Li.ui, Li.MKtoΩi, Li.Ui)
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
"""
struct OnionLocalScatteringOpImplicit <: LocalScatteringOp
    # Local lifting operator xi → ui = Li xi
    Li::LocalLiftingOp
    # Matrix corresponding to the equation (on Σ)
    # a(uΣ,uΣ^t) - ik t(uΣ,uΣ^t) = -<x,uΣ^t>
    KΣi::SparseMatrixCSC{Complex{Float64},Int64}
    # Memory allocation
    si::Vector{Complex{Float64}}
end
function OnionLocalScatteringOpImplicit(Li, KΣi)
    si = zeros(Complex{Float64}, size(KΣi,1))
    return OnionLocalScatteringOpImplicit(Li, KΣi, si)
end
function (Si::OnionLocalScatteringOpImplicit)(xi)
    ui = Si.Li(xi)
    # si = -xi - 2*Si.KΣi*ui
    mul!(Si.si, Si.KΣi, ui)
    lmul!(-2, Si.si)
    axpy!(-1, xi, Si.si)
    return Si.si
end

struct OnionLocalScatteringOpExplicit <: LocalScatteringOp
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
function OnionLocalScatteringOpExplicit(Λi, Ti, Li)
    Si = (Λi - conj.(Ti)) * inv(Λi + Ti)
    si = zeros(Complex{Float64}, size(Si,1))
    return OnionLocalScatteringOpExplicit(Λi, Ti, Li, Si, si)
end
function (Si::OnionLocalScatteringOpExplicit)(xi)
    Si.Li(xi) # To store local solution
    mul!(Si.si, Si.Si, xi)
end


"""
This local exchange op takes the form of (for 2 domains, no cross-points)

     0  -M1
    -M2  0
"""
struct OnionLocalExchangeOp <: LocalExchangeOp
    # Local part of exchange matrix
    Πi::SparseMatrixCSC{Float64,Int64}
    # Memory allocation
    Πxi::Vector{Complex{Float64}}
end
function OnionLocalExchangeOp(gid, pbs, i, MΣitoΣmt, weights)
    MΣitoΣst_s = [mapping_from_global_indices(indices_transmission_boundary(gid, pb),
                                              indices_skeleton(gid)) for pb in pbs]
    # Local part of projection operator
    PΣsttoΣi = matrix(transpose(MΣitoΣst_s[i])) * spdiagm(0=>weights)
    Pi = hcat([PΣsttoΣi * matrix(MΣitoΣst)
               for MΣitoΣst in MΣitoΣst_s]...)
    # Local part of symmetry operator
    Πi = matrix(transpose(MΣitoΣmt)) - 2 * Pi
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return OnionLocalExchangeOp(Πi, Πxi)
end
(Πi::OnionLocalExchangeOp)(x) = mul!(Πi.Πxi, Πi.Πi, x)


function OnionLocalData(pbs::Vector{P}, gid::AbstractInputData, dd::OnionDDM,
                        i::Integer;
                        to=missing) where P <: AbstractProblem
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Onion local data" begin
        pb = pbs[i]
        @timeit to "Mappings" begin
            # Indices
            ind_Ω = indices_full_domain(gid)
            ind_Σ = indices_skeleton(gid)
            ind_Ωi = indices_domain(gid, pb)
            ind_Σi = indices_transmission_boundary(gid, pb)
            NΩi = length(ind_Ωi)
            NΣi = length(ind_Σi)
            NΣks = size_multi_trace(gid)
            @info "   --> #DOF volume   $(NΩi)"
            @info "   --> #DOF surface  $(NΣi)"
            # Mappings
            MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
            MΩitoΩ = mapping_from_global_indices(ind_Ωi, ind_Ω)
            MΣitoΩi = mapping_from_global_indices(ind_Σi, ind_Ωi)
            MΣitoΣst = mapping_from_global_indices(ind_Σi, ind_Σ)
            I = sum([NΣk for (k, NΣk) in enumerate(NΣks) if k<i]) .+ (1:NΣi)
            MΣitoΣmt = Mapping(1:NΣi, I, (sum(NΣks), NΣi))
            # Correction matrix used when computing lifting
            corr = transpose(MΩitoΩ) * (1 ./ dof_weights(gid))
        end
        @timeit to "Linear system" begin
            # Local volume matrices (with and without TC) and RHS
            @timeit to "Matrix" K = get_matrix(gid, pb)
            @timeit to "Matrix" Ktilde = get_matrix_no_transmission_BC(gid, pb)
            @timeit to "RHS" fi = get_rhs(gid, pb)
            # Taking care of potential auxiliary equations
            Nfi = size(fi,1)
            NK = size(K,1)
            MKtofi = Mapping(1:Nfi, 1:Nfi, (Nfi, NK))
            MKtoΩi = Mapping(1:NΩi, 1:NΩi, (NΩi, NK))
            # Factorization and restriction of local matrices
            @timeit to "Factorization" Kinv = factorize(K)
            KΣi = Ktilde[MΣitoΩi.lind_t,1:NΩi]
            # Local lifting of sourceysy
            Fi = MKtoΩi * (Kinv \ (transpose(MKtofi) * fi))
            # Local RHS
            fiΣi = transpose(MΣitoΩi) * (MKtoΩi * (transpose(MKtofi) * fi))
            bi = - 2 * KΣi * Fi + 2 * fiΣi
        end
        @timeit to "Lifting operator" begin
            Li = OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
        end
        @timeit to "Scattering operator" begin
            if dd.implicit
                Si = OnionLocalScatteringOpImplicit(Li, KΣi)
            else
                Λi = DtN(gid, pb)
                Ti = get_transmission_matrix(gid, pb)
                Si = OnionLocalScatteringOpExplicit(Λi, Ti, Li)
            end
        end
        @timeit to "Exchange operator" begin
            weights = MΩtoΣ * (1 ./ dof_weights(gid))
            Πi = OnionLocalExchangeOp(gid, pbs, i, MΣitoΣmt, weights)
        end
    end
    return OnionLocalData(MΩitoΩ, MΣitoΣmt, corr, Li, Si, Πi, Fi, bi, to)
end


localdata_type(dd::OnionDDM) = OnionLocalData
globaldata_type(dd::OnionDDM) = OnionGlobalData
