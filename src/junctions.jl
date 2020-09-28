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
    MΩitoΩ::SparseMatrixCSC{Bool,Int64}
    # Mapping from local trace vector to global trace vector Σi → ΣMT
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Correction matrix used when computing lifting
    Ctbc_Σi::SparseMatrixCSC{Float64,Int64}
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
struct JunctionsLocalLiftingOp{T} <: LocalLiftingOp where T
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
function JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti)
    xi = zeros(Complex{Float64},size(MΣitoΩi,2))
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, Ti, xi, fi, Fi, Ui, ui)
end
function JunctionsLocalLiftingOp(MΣitoΩi, Kinv, Ti)
    xi = zeros(Complex{Float64},size(MΣitoΩi,2))
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},0)
    Ui = zeros(Complex{Float64},0)
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    JunctionsLocalLiftingOp(MΣitoΩi, 1, Kinv, Ti, xi, fi, Fi, Ui, ui)
end
function (Li::JunctionsLocalLiftingOp{SparseMatrixCSC{Bool,Int64}})(xi)
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
function (Li::JunctionsLocalLiftingOp{Int64})(xi)
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
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Mapping Σi to Σ single-trace
    MΣitoΣst::SparseMatrixCSC{Bool,Int64}
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
function JunctionsLocalExchangeOpExplicit(pbs::Vector{P}, i, MΣitoΣst, MΣitoΣmt, Ti) where P <: AbstractProblem
    # Memory allocation of projection operator
    Πi = zeros(Complex{Float64},size(transpose(MΣitoΣmt)))
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return JunctionsLocalExchangeOpExplicit(MΣitoΣmt, MΣitoΣst, Ti, Πi, Πxi)
end
(Πi::JunctionsLocalExchangeOpExplicit)(x) = mul!(Πi.Πxi, Πi.Πi, x)


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
function setup_exchange(gd::JunctionsGlobalData,
                        lds::Vector{JunctionsLocalData}, x)
    if typeof(lds[1]) <: JunctionsLocalData{<:LocalLiftingOp,<:LocalScatteringOp,<:JunctionsLocalExchangeOpImplicit}
        # Computation of RHS
        lmul!(0, gd.bst)
        for ld in lds # TODO parallelize
            Πi = ld.Πi
            mul!(Πi.ai, transpose(Πi.MΣitoΣmt), x)
            mul!(Πi.bi, Πi.Ti, Πi.ai)
            mul!(gd.tmpst, Πi.MΣitoΣst, Πi.bi)
            axpy!(1, gd.tmpst, gd.bst)
        end
        # Solve problem
        lmul!(0, gd.xst)
        if length(gd.ProjP) == 0 # No preconditioning
            gd.xst, ch = cg!(gd.xst, gd.ProjA, gd.bst;
                            tol=1.e-14, verbose=true, log=true)
        else
            gd.xst, ch = cg!(gd.xst, gd.ProjA, gd.bst; Pl= gd.ProjP.f,
                            tol=1.e-14, verbose=true, log=true)
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
    @timeit to "Local data" begin
        @info "   --> #DOF volume   $(number_of_elements(gid.m,pbs[i].Ω,dofdim(pbs[i])))"
        @info "   --> #DOF surface  $(number_of_elements(gid.m,transmission_boundary(pbs[i]),dofdim(pbs[i])))"
        @timeit to "Mappings" begin
            pb = pbs[i]
            # Definition of some restriction matrices
            @time RΩi = restriction(gid.m,pb.Ω,dofdim(pb))
            @time RΣi = restriction(gid.m,transmission_boundary(pb),dofdim(pb))
            # Using restriction matrices
            @time MΣitoΩi = RΩi * transpose(RΣi)
            @time MΩitoΩ = gid.RΩ * transpose(RΩi)
            # Mapping Σi to Σ multi-trace
            @time MΣitoΣmt = mapping_ΣitoΣmt(gid.NΣis, i)
            # Mapping Σi to Σ single-trace
            @time MΣitoΣst = gid.RΣ * transpose(RΣi)
            # Correction matrix used when computing lifting
            @time Cwsi = transpose(MΣitoΣst) * gid.Cwsst * MΣitoΣst
            @time cwi,cwj,cwv = findnz(Cwsi)
            @time IdΣi = sparse(cwi,cwj,one.(cwv),size(Cwsi)...)
            @time Ctbc_Σi = MΣitoΩi * (IdΣi - Cwsi) * transpose(MΣitoΩi)
        end
        @timeit to "Linear system" begin
            # Local volume matrix
            K = get_matrix(gid.m,pb,RΩi;to=to)
            @timeit to "Factorization" Kinv = factorize(K)
            # Local volume RHS
            fi = get_rhs(gid.m,pb;to=to)
            # Taking care of potential auxiliary equations
            extension = number_of_elements(gid.m,pb.Ω,dofdim(pb)) != size(K,1)
            MKtoΩi = extension ? sparse(I, number_of_elements(gid.m,pb.Ω,dofdim(pb)), size(K,1)) : sparse(I,0,0)
            MKtofi = extension ? sparse(I, size(fi,1), size(K,1)) : sparse(I,0,0)
            # Local lifting of sources
            @timeit to "F/B substitutions" begin
                if extension
                    Fi = MKtoΩi * (Kinv \ (transpose(MKtofi) * fi))
                else
                    Fi = Kinv \ fi
                end
            end
            # Local RHS
            bi = 2 * im * transpose(MΣitoΩi) * Fi
        end
        @timeit to "Lifting operator" begin
            tbc = [bc for bc in pb.BCs if typeof(bc)<:TransmissionBC]
            @assert length(tbc) == 1
            Ti = matrix(gid.m,pb,tbc[1])
            if extension
                Li = JunctionsLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, pb.medium.k0 .* Ti)
            else
                Li = JunctionsLocalLiftingOp(MΣitoΩi, Kinv, pb.medium.k0 .* Ti)
            end
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
                              typeof(Πi)}(MΩitoΩ, MΣitoΣmt, Ctbc_Σi, Li, Si,
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
                PP = STProjector_Precond(lds, Nst, gid.Cwsst)
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
            # Mapping Σ single-trace to Σ mutli-trace
            MΣSTtoΣMT = sum([ld.Πi.MΣitoΣmt * transpose(ld.Πi.MΣitoΣst)
                            for ld in lds])
            # Transmission matrix MT tested by ST
            T0MT_STt = sum([ld.Πi.MΣitoΣst * ld.Πi.Ti * transpose(ld.Πi.MΣitoΣmt)
                            for ld in lds])
            # Transmission matrix ST tetsed by ST
            T0ST_STt = sum([ld.Πi.MΣitoΣst * ld.Πi.Ti * transpose(ld.Πi.MΣitoΣst)
                            for ld in lds])
            # Violent inversion of matrix... of course very inefficient...
            P0 = MΣSTtoΣMT * inv(Matrix(T0ST_STt)) * T0MT_STt
            # Local storage: Π = 2 P - I
            for ld in lds
                ld.Πi.Πi = transpose(ld.Πi.MΣitoΣmt) * 2 * P0 - transpose(ld.Πi.MΣitoΣmt)
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
