init_global_solution(ld::AbstractLocalData) = zeros(Complex{Float64},size(ld.MΩitoΩ,1))
init_local_solution(ld::AbstractLocalData) = zeros(Complex{Float64},size(ld.MΩitoΩ,2))
init_global_trace(ld::AbstractLocalData) = zeros(Complex{Float64},size(ld.MΣitoΣmt,1))
init_local_trace(ld::AbstractLocalData) = zeros(Complex{Float64},size(ld.MΣitoΣmt,2))
global2local_trace(ld::AbstractLocalData, x) = transpose(ld.MΣitoΣmt) * x
global2local_trace(ld::AbstractLocalData, x, xi) = mul!(xi, transpose(ld.MΣitoΣmt), x)
local2global_solution(ld::AbstractLocalData, ui) = ld.MΩitoΩ * ui
local2global_solution(ld::AbstractLocalData, ui, u) = mul!(u, ld.MΩitoΩ, ui)
local2global_trace(ld::AbstractLocalData, xi) = ld.MΣitoΣmt * xi
local2global_trace(ld::AbstractLocalData, xi, x) = mul!(x, ld.MΣitoΣmt, xi)
local_lifting_correction(ld::AbstractLocalData, ui) = ld.corr .* ui
function local_lifting_correction(ld::AbstractLocalData, ui, vi)
    vi .= ld.corr .* ui
end


"""
Global operators are just wrappers around global and local data.
"""

struct GlobalLiftingOp{TG<:AbstractGlobalData,TL<:AbstractLocalData}
    gd::TG
    lds::Array{TL,1}
    # Memory allocation
    u::Array{Complex{Float64},1}            # Global solution
    v::Array{Complex{Float64},1}            # Global solution
    vis::Array{Array{Complex{Float64},1},1} # Local solutions with correction
    xis::Array{Array{Complex{Float64},1},1} # Local traces
end
function GlobalLiftingOp(gd, lds)
    u = init_global_solution(lds[1])
    v = init_global_solution(lds[1])
    vis = [init_local_solution(ld) for ld in lds]
    xis = [init_local_trace(ld) for ld in lds]
    GlobalLiftingOp(gd,lds,u,v,vis,xis)
end
function (GL::GlobalLiftingOp)(x)
    lmul!(0,GL.u)
    for (xi, vi, ld) in zip(GL.xis, GL.vis, GL.lds) # TODO parallelize
        global2local_trace(ld, x, xi)
        ui = ld.Li(xi)
        local_lifting_correction(ld, ui, vi)
        lmul!(0,GL.v)
        local2global_solution(ld, vi, GL.v)
        axpy!(1, GL.v, GL.u)
    end
    return GL.u
end

struct GlobalScatteringOp{TG<:AbstractGlobalData,TL<:AbstractLocalData}
    gd::TG
    lds::Array{TL,1}
    # Memory allocation
    y::Array{Complex{Float64},1}            # Global multi trace
    s::Array{Complex{Float64},1}            # Global multi trace
    xis::Array{Array{Complex{Float64},1},1} # Local traces
end
function GlobalScatteringOp(gd, lds)
    y = init_global_trace(lds[1])
    s = init_global_trace(lds[1])
    xis = [init_local_trace(ld) for ld in lds]
    GlobalScatteringOp(gd,lds,y,s,xis)
end
function (GS::GlobalScatteringOp)(x)
    lmul!(0,GS.y)
    for (xi, ld) in zip(GS.xis, GS.lds) # TODO parallelize
        global2local_trace(ld, x, xi)
        si = ld.Si(xi)
        local2global_trace(ld, si, GS.y)
    end
    return GS.y
end

# Default setup step
setup_exchange(gd, lds, x) = nothing

struct GlobalExchangeOp{TG<:AbstractGlobalData,TL<:AbstractLocalData}
    gd::TG
    lds::Array{TL,1}
    # Memory allocation
    y::Array{Complex{Float64},1}            # Global multi trace
    s::Array{Complex{Float64},1}            # Global multi trace
end
function GlobalExchangeOp(gd, lds)
    y = init_global_trace(lds[1])
    s = init_global_trace(lds[1])
    GlobalExchangeOp(gd,lds,y,s)
end
function (Π::GlobalExchangeOp)(x)
    lmul!(0,Π.y)
    # Setup
    setup_exchange(Π.gd, Π.lds, x)
    for ld in Π.lds # TODO parallelize
        Πxi = ld.Πi(x)
        lmul!(0,Π.s)
        local2global_trace(ld, Πxi, Π.s)
        axpy!(1, Π.s, Π.y)
    end
    return Π.y
end

struct Aop
    # Scattering operator
    S::GlobalScatteringOp
    # Exchange operator
    Π::GlobalExchangeOp
    # Timer
    to::TimerOutput
end
function (A::Aop)(x)
    @timeit A.to "Scattering" Sx = A.S(x)
    @timeit A.to "Exchange" Ax = A.Π(Sx)
    return Ax
end


"""
Global struct that contains all information to perform the DDM.
"""
struct DDM{T1,T2,R<:AbstractGlobalData,S<:AbstractLocalData}
    # Global data
    gd::R
    # Local data
    lds::Array{S,1}
    # Iterative operator (I-A)x = b
    A::Aop
    # Lifting operator without physical source
    L::GlobalLiftingOp
    # Solution with physical source but 0 transmission BCs: u = Lx + F
    F::Array{T1,1} # vector of DOF
    # RHS of interface problem (I-A)x = b
    b::Array{T2,1} # expressed in weak form <ik (Tij+Tji) Fi, G^t>
end
function DDM(pbs::Vector{P}, gid::I, dd::T; to=missing
            ) where P <: AbstractProblem where I <: AbstractInputData where T <: DDM_Type
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Offline" begin
        n = length(pbs)
        if length(pbs) < 2 @warn "Number of domains < 2." end
        @info "==> Number of Problems $n"
        # Offline local computations
        lds = Vector{localdata_type(dd)}(undef, n)
        for i in 1:n # TODO parallelize
            @info "==> Problem $i on CPU $(myid())"
            lds[i] = localdata_type(dd)(pbs, gid, dd, i; to=to)
        end
        # Offline global computations
        gd = globaldata_type(dd)(lds, gid, dd; to=to)
        # Global operators
        L = GlobalLiftingOp(gd, lds)
        S = GlobalScatteringOp(gd, lds)
        Π = GlobalExchangeOp(gd, lds)
        A = Aop(S,Π,to)
        # Taking care of source terms
        F = init_global_solution(lds[1])
        b = init_global_trace(lds[1])
        for ld in lds # Reduction of local source terms
            Fi = local_lifting_correction(ld, ld.Fi)
            F += local2global_solution(ld, Fi)
            b += local2global_trace(ld, ld.bi)
        end
        b = copy(Π(b))
    end
    DDM(gd,lds,A,L,F,b)
end
