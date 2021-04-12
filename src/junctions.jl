"""
Contains DDM parameters, based on exchange operator constructed from
a projection.
"""
struct JunctionsDDM <: DDM_Type
    implicit::Bool
    precond::Bool
    JunctionsDDM(;implicit=true, precond=true) = new(implicit, precond)
end


localdata_type(dd::JunctionsDDM) = JunctionsLocalData
globaldata_type(dd::JunctionsDDM) = JunctionsGlobalData


mutable struct JunctionsGlobalData <: AbstractGlobalData
    # Global projection operator and preconditionner
    ProjA::LinearMap{Float64}
    ProjP::LinearMap{Float64}
    # Log for min, max, sum (to compute mean) inner cg iterations
    cg_min::Int64
    cg_max::Int64
    cg_sum::Int64
    # Memory allocation
    xst::Vector{Complex{Float64}}
    bst::Vector{Complex{Float64}}
    tmpst::Vector{Complex{Float64}}
end


struct JunctionsLocalData{TL<:LocalLiftingOp, TS<:LocalScatteringOp,
                          TP<:LocalExchangeOp} <: AbstractLocalData
    # Local to global solution mappings Ωi → Ω
    MΩitoΩ::Mapping
    # Mapping from local trace vector to global trace vector Σi → ΣMT
    MΣitoΣmt::Mapping
    # Correction matrix used when computing lifting
    corr::Vector{Float64}
    Li::TL
    Si::TS
    Πi::TP
    Fi::Array{Complex{Float64},1}
    bi::Array{Complex{Float64},1}
    # Timer
    to::TimerOutput
end


"""
Local lifting operator: takes a trace and computes a local solution.

It is parametrized with T for efficiency reasons.
"""
struct JunctionsLocalLiftingOp <: LocalLiftingOp
    # Mapping from trace vector to local problem rhs vector Σi → Ωi
    MΣitoΩi::Mapping
    # Mapping from (possibly extended) solution of K to DOFs of Ω
    MKtoΩi::Mapping
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
function JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti)
    xi = zeros(Complex{Float64},size(MΣitoΩi,2))
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti, xi, fi, Fi, Ui, ui)
end
function (Li::JunctionsLocalLiftingOp)(xi)
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

"""
Local scattering operator.
"""
struct JunctionsLocalScatteringOp <: LocalScatteringOp
    # Local lifting operator xi → ui = Li xi
    Li::JunctionsLocalLiftingOp
    # Memory allocation
    si::Vector{Complex{Float64}}
end
function JunctionsLocalScatteringOp(Li)
    si = zeros(Complex{Float64}, size(Li.MΣitoΩi,2))
    return JunctionsLocalScatteringOp(Li, si)
end
function (Si::JunctionsLocalScatteringOp)(xi)
    ui = Si.Li(xi)
    # si = -xi + 2im * transpose(Si.Li.MΣitoΩi) * ui
    mul!(Si.si, transpose(Si.Li.MΣitoΩi), ui)
    lmul!(2im, Si.si)
    axpy!(-1, xi, Si.si)
    return Si.si
end


mutable struct JunctionsLocalExchangeOpImplicit <: LocalExchangeOp
    # Mapping Σi to Σ multi-trace
    MΣitoΣmt::Mapping
    # Mapping Σi to Σ single-trace
    MΣitoΣst::Mapping
    # Transmission operator
    Ti::Matrix{Complex{Float64}}
    # Local part of projection operator
    invTi::Matrix{Complex{Float64}}
    # Memory allocation
    pxi::Vector{Complex{Float64}} # Solution local projection
    Πxi::Vector{Complex{Float64}} # Solution local exchange
    ai::Vector{Complex{Float64}} # Local allocation matvec and precond
    bi::Vector{Complex{Float64}} # Local allocation matvec and precond
end
function JunctionsLocalExchangeOpImplicit(pbs, i, MΣitoΣst, MΣitoΣmt, Ti)
    # Factorisation of Ti
    invTi = inv(Matrix(Ti))
    # Memory allocation
    NΣi = size(MΣitoΣst,2)
    pxi = zeros(Complex{Float64}, NΣi)
    Πxi = zeros(Complex{Float64}, NΣi)
    ai = zeros(Complex{Float64}, NΣi)
    bi = zeros(Complex{Float64}, NΣi)
    return JunctionsLocalExchangeOpImplicit(MΣitoΣmt, MΣitoΣst, Ti, invTi, pxi,
                                            Πxi, ai, bi)
end
function (Πi::JunctionsLocalExchangeOpImplicit)(x)
    # 2 .* Πi.pxi .- Πi.MΣitoΣmt * x
    mul!(Πi.Πxi, transpose(Πi.MΣitoΣmt), x)
    lmul!(-1, Πi.Πxi)
    axpy!(2, Πi.pxi, Πi.Πxi)
    return Πi.Πxi
end

mutable struct JunctionsLocalExchangeOpExplicit <: LocalExchangeOp
    # Mapping Σi to Σ multi-trace
    MΣitoΣmt::Mapping
    # Mapping Σi to Σ single-trace
    MΣitoΣst::Mapping
    # Transmission operator
    Ti::Matrix{Complex{Float64}}
    # Local part of projection operator
    Πi::Matrix{Complex{Float64}}
    # Memory allocation
    Πxi::Vector{Complex{Float64}}
end
function JunctionsLocalExchangeOpExplicit(pbs::Vector{P}, i, MΣitoΣst, MΣitoΣmt, Ti) where P <: AbstractProblem
    # Memory allocation of projection operator
    Πi = zeros(Complex{Float64},size(transpose(MΣitoΣmt)))
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return JunctionsLocalExchangeOpExplicit(MΣitoΣmt, MΣitoΣst, Ti, Πi, Πxi)
end
(Πi::JunctionsLocalExchangeOpExplicit)(x) = mul!(Πi.Πxi, Πi.Πi, x)


struct STProjector
    lds::Vector{JunctionsLocalData}
    # Memory allocation
    tmpst::Vector{Complex{Float64}}
end
function STProjector(lds::Vector{JunctionsLocalData}, Nst::Int64)
    # Memory allocation
    tmpst = zeros(Complex{Float64}, Nst)
    return STProjector(lds, tmpst)
end
function (P::STProjector)(y, x)
    fill!(y, 0)
    for ld in P.lds # TODO parallelize
        Πi = ld.Πi
        #y += Πi.MΣitoΣst * Πi.Ti * transpose(Πi.MΣitoΣst) * x
        mul!(Πi.ai, transpose(Πi.MΣitoΣst), x)
        mul!(Πi.bi, Πi.Ti, Πi.ai)
        fill!(P.tmpst, 0)
        mul!(P.tmpst, Πi.MΣitoΣst, Πi.bi)
        axpy!(1, P.tmpst, y)
    end
    return y
end

struct STProjector_Precond
    lds::Vector{JunctionsLocalData}
    # Weights at transmission DOFs
    weights::Vector{Float64}
    # Memory allocation
    yst::Vector{Complex{Float64}}
    cxst::Vector{Complex{Float64}}
    tmpst::Vector{Complex{Float64}}
end
function STProjector_Precond(lds::Vector{JunctionsLocalData}, Nst::Int64, weights)
    # Memory allocation
    yst = zeros(Complex{Float64}, Nst)
    cxst = zeros(Complex{Float64}, Nst)
    tmpst = zeros(Complex{Float64}, Nst)
    return STProjector_Precond(lds, weights, yst, cxst, tmpst)
end
function ldiv!(y, P::STProjector_Precond, x)
    fill!(P.yst, 0)
    P.cxst .= P.weights .* x
    for ld in P.lds # TODO parallelize
        Πi = ld.Πi
        #y += P.weights * Πi.MΣitoΣst * (Πi.invTi * (transpose(Πi.MΣitoΣst) * P.weights * x))
        mul!(Πi.ai, transpose(Πi.MΣitoΣst), P.cxst)
        mul!(Πi.bi, Πi.invTi, Πi.ai)
        fill!(P.tmpst, 0)
        mul!(P.tmpst, Πi.MΣitoΣst, Πi.bi)
        axpy!(1, P.tmpst, P.yst)
    end
    y .= P.weights .* P.yst
    return y
end


"""
Solving projection problem by using preconditionned CG method.
"""
function setup_exchange(gd::JunctionsGlobalData,
                        lds::Vector{JunctionsLocalData}, x)
    if typeof(lds[1]) <: JunctionsLocalData{<:LocalLiftingOp,<:LocalScatteringOp,<:JunctionsLocalExchangeOpImplicit}
        # Computation of RHS
        fill!(gd.bst, 0)
        for ld in lds # TODO parallelize
            Πi = ld.Πi
            mul!(Πi.ai, transpose(Πi.MΣitoΣmt), x)
            mul!(Πi.bi, Πi.Ti, Πi.ai)
            fill!(gd.tmpst, 0)
            mul!(gd.tmpst, Πi.MΣitoΣst, Πi.bi)
            axpy!(1, gd.tmpst, gd.bst)
        end
        # Solve problem
        fill!(gd.xst, 0)
        if length(gd.ProjP) == 0 # No preconditioning
            gd.xst, ch = cg!(gd.xst, gd.ProjA, gd.bst;
                            reltol=1.e-14, verbose=false, log=true)
        else
            gd.xst, ch = cg!(gd.xst, gd.ProjA, gd.bst; Pl= gd.ProjP.f,
                            reltol=1.e-14, verbose=false, log=true)
        end
        # Logging
        if ch.iters > 0 gd.cg_min = min(gd.cg_min, ch.iters) end
        gd.cg_max = max(gd.cg_max, ch.iters)
        gd.cg_sum += ch.iters
        # Local storage
        for ld in lds
            Πi = ld.Πi
            mul!(Πi.pxi, transpose(Πi.MΣitoΣst), gd.xst)
        end
    end
end


function JunctionsLocalData(pbs::Vector{P}, gid::AbstractInputData, dd::JunctionsDDM,
                            i::Integer;
                            to=missing) where P <: AbstractProblem
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Junctions local data" begin
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
            # Factorization of local matrix
            @timeit to "Factorization" Kinv = factorize(K)
            # Local lifting of sources
            Fi = MKtoΩi * (Kinv \ (transpose(MKtofi) * fi))
            # Local RHS
            bi = 2 * im * (transpose(MΣitoΩi) * Fi)
        end
        @timeit to "Lifting operator" begin
            Ti = get_transmission_matrix(gid, pb)
            Li = JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti)
        end
        @timeit to "Scattering operator" begin
            Si = JunctionsLocalScatteringOp(Li)
        end
        @timeit to "Exchange operator" begin
            if dd.implicit
                Πi = JunctionsLocalExchangeOpImplicit(pbs, i, MΣitoΣst, MΣitoΣmt, Li.Ti)
            else
                Πi = JunctionsLocalExchangeOpExplicit(pbs, i, MΣitoΣst, MΣitoΣmt, Li.Ti)
            end
        end
    end
    return JunctionsLocalData{typeof(Li),
                              typeof(Si),
                              typeof(Πi)}(MΩitoΩ, MΣitoΣmt, corr, Li, Si,
                                          Πi, Fi, bi, to)
end


function JunctionsGlobalData(lds::Vector{JunctionsLocalData},
                             gid::AbstractInputData, dd::JunctionsDDM;
                             to=missing)
    @timeit to "Exchange operator" begin
        # Dummy initialisation of global projection operator and preconditionner
        ProjA = LinearMap{Float64}(x->x, 0)
        ProjP = LinearMap{Float64}(x->x, 0)
        if dd.implicit
            if dd.precond
                # Operator of linear system associated to projection problem
                Nst = size(lds[1].Πi.MΣitoΣst, 1)
                PA = STProjector(lds, Nst)
                ProjA = LinearMap{Float64}(PA, Nst;
                                           ismutating=true, issymmetric=true,
                                           ishermitian=false, isposdef=true)
                # Preconditionner of linear system associated to projection problem
                ind_Ω = indices_full_domain(gid)
                ind_Σ = indices_skeleton(gid)
                MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
                weights = MΩtoΣ * (1 ./ dof_weights(gid))
                PP = STProjector_Precond(lds, Nst, weights)
                ProjP = LinearMap{Float64}(PP, Nst;
                                           ismutating=true, issymmetric=true,
                                           ishermitian=false, isposdef=true)
            else
                # Operator of linear system associated to projection problem
                Nst = size(lds[1].Πi.MΣitoΣst, 1)
                PA = STProjector(lds, Nst)
                ProjA = LinearMap{Float64}(PA, Nst;
                                           ismutating=true, issymmetric=true,
                                           ishermitian=false, isposdef=true)
            end
        else
            Tis = [ld.Πi.Ti for ld in lds]
            PΣitoΣmts = [matrix(ld.Πi.MΣitoΣmt) for ld in lds]
            PΣitoΣsts = [matrix(ld.Πi.MΣitoΣst) for ld in lds]
            # Mapping Σ single-trace to Σ mutli-trace
            MΣSTtoΣMT = sum([PΣitoΣmt * transpose(PΣitoΣst)
                             for (PΣitoΣmt, PΣitoΣst) in zip(PΣitoΣmts, PΣitoΣsts)])
            # Transmission matrix MT tested by ST
            T0MT_STt = sum([PΣitoΣst * Ti * transpose(PΣitoΣmt)
                            for (PΣitoΣmt, PΣitoΣst, Ti) in zip(PΣitoΣmts, PΣitoΣsts, Tis)])
            # Transmission matrix ST tetsed by ST
            T0ST_STt = sum([PΣitoΣst * Ti * transpose(PΣitoΣst)
                            for (PΣitoΣst, Ti) in zip(PΣitoΣsts, Tis)])
            # Violent inversion of matrix... of course very inefficient...
            P0 = MΣSTtoΣMT * inv(Matrix(T0ST_STt)) * T0MT_STt
            # Local storage: Π = 2 P - I
            for (PΣitoΣmt, ld) in zip(PΣitoΣmts, lds)
                ld.Πi.Πi = transpose(PΣitoΣmt) * 2 * P0 - transpose(PΣitoΣmt)
            end
        end
        # Memory allocation
        NΣst = size(lds[1].Πi.MΣitoΣst,1)
        xst = zeros(Complex{Float64}, NΣst)
        bst = zeros(Complex{Float64}, NΣst)
        tmpst = zeros(Complex{Float64}, NΣst)
    end
    return JunctionsGlobalData(ProjA, ProjP, 2^63-1, 0, 0, xst, bst, tmpst)
end
