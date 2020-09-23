####################
# Junction weights #
####################

function detect_junctions(m::Mesh, Ωs::Vector{Domain}, d::Int64)
    # Initialisation
    eltInNdomains = zeros(UInt16, number_of_elements(m,d))
    # Loop on domains
    for ω in Ωs
        eltInNdomains[element_indices(m,ω,d)] .+= 1
    end
    # Junctions
    bool_j = eltInNdomains .>= 2
    indices = (1:number_of_elements(m,d))[bool_j] # DOF indices
    weights = eltInNdomains[bool_j]
    return indices, weights
end


function junction_weights(m::Mesh, Ωs::Vector{Domain}, pb)
    # Detection of junction points
    ind, weights = detect_junctions(m, Ωs, dofdim(pb))
    # Initialisation
    N = number_of_elements(m, union(Ωs...), dofdim(pb))
    # Treatment of junction points
    Cws = sparse(ind, ind, 1 ./ weights, N, N)
    return Cws
end


##############################
# Offline local computations #
##############################

struct GlobalOfflineData
    Σ::Domain
    RΩ::SparseMatrixCSC{Bool,Int64}
    RΣ::SparseMatrixCSC{Bool,Int64}
    Cwsst::SparseMatrixCSC{Float64,Int64}
    NΣis::Vector{Int64}
end
function GlobalOfflineData(m::Mesh, fullpb::P, pbs::Vector{P};
                           to=missing, kwargs...) where P <: AbstractProblem
    @timeit to "Global data" begin
        @info "   --> #DOF volume   $(number_of_elements(m,fullpb.Ω,dofdim(fullpb)))"
        @info "   --> #DOF skeleton $(number_of_elements(m,skeleton(fullpb.Ω),dofdim(fullpb)))"
        # Skeleton
        Σ = union(Domain.(unique(vcat([transmission_boundary(pb)[:] for pb in pbs]...)))...)
        # Definition of some restriction matrices
        @timeit to "RΩ" RΩ = restriction(m,fullpb.Ω,dofdim(fullpb))
        @timeit to "RΣ" RΣ = restriction(m,Σ,dofdim(fullpb))
        # Diagonal matrix of junction weights at transmission DOFs
        @timeit to "Junction weights" Cws = junction_weights(m, [p.Ω for p in pbs], fullpb)
        @timeit to "Cwsst" Cwsst = RΣ * Cws * transpose(RΣ)
        # Size
        @timeit to "NΣis" NΣis = [number_of_elements(m,transmission_boundary(pb),dofdim(pb)) for pb in pbs]
    end
    GlobalOfflineData(Σ, RΩ, RΣ, Cwsst, NΣis)
end

abstract type FemLocalData <: AbstractLocalData end

struct LocalData{TS<:LocalScatteringOp,TΠ<:LocalExchangeOp} <: FemLocalData
    # Local to global solution mappings Ωi → Ω
    MΩitoΩ::SparseMatrixCSC{Bool,Int64}
    # Mapping from local trace vector to global trace vector Σi → ΣMT
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Correction matrix used when computing lifting
    Ctbc_Σi::SparseMatrixCSC{Float64,Int64}
    Li::LocalLiftingOp
    Si::TS
    Πi::TΠ
    Fi::Array{Complex{Float64},1}
    bi::Array{Complex{Float64},1}
    # Timer
    to::TimerOutput
end

function mapping_ΣitoΣmt(NΣis, i)
    i0 = 1 # first index of line
    imax = 0 # total number of lines
    for (k,NΣk) in enumerate(NΣis)
        imax += NΣk
        if k<i i0 += NΣk end
    end
    I = i0 : (i0+NΣis[i]-1)
    J = 1 : NΣis[i]
    V = ones(Bool,NΣis[i])
    return sparse(I,J,V,imax,NΣis[i])
end


function LocalData(m::Mesh, fullpb::P, pbs::Vector{P}, god,
                    i::Integer; mode=:implicit, exchange_type=:basic,
                    to=missing, kwargs...) where P <: AbstractProblem
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Local data" begin
        @info "   --> #DOF volume   $(number_of_elements(m,pbs[i].Ω,dofdim(pbs[i])))"
        @info "   --> #DOF surface  $(number_of_elements(m,transmission_boundary(pbs[i]),dofdim(pbs[i])))"
        @timeit to "Mappings" begin
            pb = pbs[i]
            # Definition of some restriction matrices
            @time RΩi = restriction(m,pb.Ω,dofdim(pb))
            @time RΣi = restriction(m,transmission_boundary(pb),dofdim(pb))
            # Using restriction matrices
            @time MΣitoΩi = RΩi * transpose(RΣi)
            @time MΩitoΩ = god.RΩ * transpose(RΩi)
            # Mapping Σi to Σ multi-trace
            @time MΣitoΣmt = mapping_ΣitoΣmt(god.NΣis, i)
            # Mapping Σi to Σ single-trace
            @time MΣitoΣst = god.RΣ * transpose(RΣi)
            # Correction matrix used when computing lifting
            @time Cwsi = transpose(MΣitoΣst) * god.Cwsst * MΣitoΣst
            @time cwi,cwj,cwv = findnz(Cwsi)
            @time IdΣi = sparse(cwi,cwj,one.(cwv),size(Cwsi)...)
            @time Ctbc_Σi = MΣitoΩi * (IdΣi - Cwsi) * transpose(MΣitoΩi)
        end
        @timeit to "Linear system" begin
            # Local volume matrix
            K = get_matrix(m,pb,RΩi;to=to)
            @timeit to "Factorization" Kinv = factorize(K)
            # Local volume RHS
            fi = get_rhs(m,pb;to=to)
            # Taking care of potential auxiliary equations
            extension = number_of_elements(m,pb.Ω,dofdim(pb)) != size(K,1)
            MKtoΩi = extension ? sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(K,1)) : sparse(I,0,0)
            MKtofi = extension ? sparse(I, size(fi,1), size(K,1)) : sparse(I,0,0)
            # Local lifting of sources
            @timeit to "F/B substitutions" begin
                if extension
                    Fi = MKtoΩi * (Kinv \ (transpose(MKtofi) * fi))
                else
                    Fi = Kinv \ fi
                end
            end
            if exchange_type == :xpts
                # Local RHS
                bi = 2 * im * transpose(MΣitoΩi) * Fi
            else
                # Local matrix without TBC
                pb_noTBC = typeof(pb)(pb.medium, pb.Ω, filter(bc->typeof(bc)<:PhysicalBC, pb.BCs))
                Ktilde = get_matrix(m,pb_noTBC,RΩi;to=to)
                MKtildetoΩi = sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(Ktilde,1))
                KΣi = RΣi * transpose(RΩi) * MKtildetoΩi * Ktilde * transpose(MKtildetoΩi)
                # Local RHS
                if extension
                    bi = - 2 * KΣi * Fi + 2 * transpose(MΣitoΩi) * MKtoΩi * transpose(MKtofi) * fi
                else
                    bi = - 2 * KΣi * Fi + 2 * transpose(MΣitoΩi) * fi
                end
            end
        end
        @timeit to "Lifting operator" begin
            if exchange_type == :xpts
                tbc = [bc for bc in pb.BCs if typeof(bc)<:TransmissionBC]
                @assert length(tbc) == 1
                Ti = matrix(m,pb,tbc[1])
                if extension
                    Li = XPLiftingOp(MΣitoΩi, MKtoΩi, Kinv, pb.medium.k0 .* Ti)
                else
                    Li = XPLiftingOp(MΣitoΩi, Kinv, pb.medium.k0 .* Ti)
                end
            else
                if extension
                    Li = BasicLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
                else
                    Li = BasicLiftingOp(MΣitoΩi, Kinv)
                end
            end
        end
        @timeit to "Scattering operator" begin
            if exchange_type == :xpts
                Si = XPScatteringOp(Li)
            else
                if mode == :explicit
                    Σ = remove(union([bc.Γ for bc in pb.BCs
                                    if typeof(bc) <: PhysicalBC]...),
                            boundary(pb.Ω))
                    Λi = DtN(m, pb, Σ)

                    # Computation of transmission matrix
                    Ti = spzeros(Float64,size(RΣi,1),size(RΣi,1))
                    # Loop needed in case of more than one connected transmission boundary
                    for bc in pb.BCs
                        if typeof(bc) <: TransmissionBC
                            RΣii = restriction(m, bc.Γ, dofdim(pb))
                            Ti += RΣi * transpose(RΣii) * matrix(m,pb,bc) * RΣii * transpose(RΣi)
                        end
                    end
                    Si = ExplicitLocalScatteringOp(Λi, Ti, Li)
                else
                    Si = ImplicitLocalScatteringOp(Li, KΣi)
                end
            end
        end
        @timeit to "Exchange operator" begin
            if exchange_type == :xpts
                if mode == :explicit
                    Πi = ExplicitXPExchangeOp(m, fullpb, pbs, i, MΣitoΣst, MΣitoΣmt, Li.Ti)
                else
                    Πi = ImplicitXPExchangeOp(m, fullpb, pbs, i, MΣitoΣst, MΣitoΣmt, Li.Ti)
                end
            elseif exchange_type == :basic
                Πi = BasicExchangeOp(m, fullpb, pbs, i, god.RΣ, RΣi, MΣitoΣmt, god.Cwsst)
            end
        end
    end
    return LocalData{typeof(Si), typeof(Πi)}(MΩitoΩ, MΣitoΣmt,
                                             Ctbc_Σi, Li, Si, Πi, Fi, bi, to)
end

init_global_solution(ld::FemLocalData) = zeros(Complex{Float64},size(ld.MΩitoΩ,1))
init_local_solution(ld::FemLocalData) = zeros(Complex{Float64},size(ld.MΩitoΩ,2))
init_global_trace(ld::FemLocalData) = zeros(Complex{Float64},size(ld.MΣitoΣmt,1))
init_local_trace(ld::FemLocalData) = zeros(Complex{Float64},size(ld.MΣitoΣmt,2))
global2local_trace(ld::FemLocalData, x) = transpose(ld.MΣitoΣmt) * x
global2local_trace(ld::FemLocalData, x, xi) = mul!(xi, transpose(ld.MΣitoΣmt), x)
local2global_solution(ld::FemLocalData, ui) = ld.MΩitoΩ * ui
local2global_solution(ld::FemLocalData, ui, u) = mul!(u, ld.MΩitoΩ, ui)
local2global_trace(ld::FemLocalData, xi) = ld.MΣitoΣmt * xi
local2global_trace(ld::FemLocalData, xi, x) = mul!(x, ld.MΣitoΣmt, xi)
local_lifting_correction(ld::FemLocalData, ui) = ui - ld.Ctbc_Σi * ui
function local_lifting_correction(ld::FemLocalData, ui, vi)
    # vi = ui - ld.Ctbc_Σi * ui
    mul!(vi, ld.Ctbc_Σi, ui)
    lmul!(-1, vi)
    axpy!(1, ui, vi)
end

###########
# FEM DDM #
###########

function local_offline(i::Integer,m::Mesh,fullpb::P,pbs::Vector{P},
                 lds::Array{T,1}, god; kwargs...) where T <: FemLocalData where P <: AbstractProblem
    @info "==> Problem $i on CPU $(myid())"
    lds[i] = T(m,fullpb,pbs,god,i;kwargs...)
end


function global_offline(lds::Array{<:AbstractLocalData,1}, god::GlobalOfflineData;
                        exchange_type=:basic, mode=:explicit, to=missing, kwargs...)
    if exchange_type == :xpts
        @timeit to "Exchange operator" begin
            if mode == :explicit
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
                for ld in lds ld.Πi.Πi = transpose(ld.Πi.MΣitoΣmt) * 2 * P0 - transpose(ld.Πi.MΣitoΣmt) end
            elseif mode == :implicit_noP
                # Operator of linear system associated to projection problem
                Π1 = lds[1].Πi
                Nst = size(Π1.MΣitoΣst, 1)
                PA = STProjector(lds, Nst)
                LMA = LinearMap{Float64}(PA, Nst;
                                         ismutating=true, issymmetric=true,
                                         ishermitian=false, isposdef=true)
                # Storage in first problem
                Π1.PA = LMA
            else
                # Operator of linear system associated to projection problem
                Π1 = lds[1].Πi
                Nst = size(Π1.MΣitoΣst, 1)
                PA = STProjector(lds, Nst)
                LMA = LinearMap{Float64}(PA, Nst;
                                         ismutating=true, issymmetric=true,
                                         ishermitian=false, isposdef=true)
                # Preconditionner of linear system associated to projection problem
                PP = STProjector_Precond(lds, Nst, god.Cwsst)
                LMP = LinearMap{Float64}(PP, Nst;
                                         ismutating=true, issymmetric=true,
                                         ishermitian=false, isposdef=true)
                # Storage in first problem
                Π1.PA = LMA
                Π1.PP = LMP
            end
        end
    end
end

function femDDM(m::Mesh, fullpb::P, pbs::Array{P,1}; to=missing,
                kwargs...) where P <: AbstractProblem
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Offline" begin
        n = length(pbs)
        @info "==> Number of Problems $n"
        # Offline global computation
        god = GlobalOfflineData(m, fullpb, pbs; to=to, kwargs...)
        # Offline local computations
        lds = Array{LocalData,1}(undef,n)
        for i in 1:n # TODO parallelize
            local_offline(i, m, fullpb, pbs, lds, god; to=to, kwargs...)
        end
        # Offline global computations
        global_offline(lds, god; to=to, kwargs...)
    end
    DDM(Vector{typeof(lds[1])}(lds), to)
end
