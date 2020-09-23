"""
Global operators are just wrappers around local data array.
"""

struct GlobalLiftingOp{T<:AbstractLocalData}
    lds::Array{T,1}
    # Memory allocation
    u::Array{Complex{Float64},1}            # Global solution
    v::Array{Complex{Float64},1}            # Global solution
    vis::Array{Array{Complex{Float64},1},1} # Local solutions with correction
    xis::Array{Array{Complex{Float64},1},1} # Local traces
end
function GlobalLiftingOp(lds)
    u = init_global_solution(lds[1])
    v = init_global_solution(lds[1])
    vis = [init_local_solution(ld) for ld in lds]
    xis = [init_local_trace(ld) for ld in lds]
    GlobalLiftingOp(lds,u,v,vis,xis)
end
function (GL::GlobalLiftingOp)(x)
    lmul!(0,GL.u)
    for (xi, vi, ld) in zip(GL.xis, GL.vis, GL.lds) # TODO parallelize
        global2local_trace(ld, x, xi)
        ui = ld.Li(xi)
        local_lifting_correction(ld, ui, vi)
        local2global_solution(ld, vi, GL.v)
        axpy!(1, GL.v, GL.u)
    end
    return GL.u
end

struct GlobalScatteringOp{T<:AbstractLocalData}
    lds::Array{T,1}
    # Memory allocation
    y::Array{Complex{Float64},1}            # Global multi trace
    s::Array{Complex{Float64},1}            # Global multi trace
    xis::Array{Array{Complex{Float64},1},1} # Local traces
end
function GlobalScatteringOp(lds)
    y = init_global_trace(lds[1])
    s = init_global_trace(lds[1])
    xis = [init_local_trace(ld) for ld in lds]
    GlobalScatteringOp(lds,y,s,xis)
end
function (GS::GlobalScatteringOp)(x)
    lmul!(0,GS.y)
    for (xi, ld) in zip(GS.xis, GS.lds) # TODO parallelize
        global2local_trace(ld, x, xi)
        si = ld.Si(xi)
        local2global_trace(ld, si, GS.s)
        axpy!(1, GS.s, GS.y)
    end
    return GS.y
end

# Default setup step
setup_exchange(lds, x) = nothing

struct ExchangeOp{T<:AbstractLocalData}
    lds::Array{T,1}
    # Memory allocation
    y::Array{Complex{Float64},1}            # Global multi trace
    s::Array{Complex{Float64},1}            # Global multi trace
end
function ExchangeOp(lds)
    y = init_global_trace(lds[1])
    s = init_global_trace(lds[1])
    ExchangeOp(lds,y,s)
end
function (Π::ExchangeOp)(x)
    lmul!(0,Π.y)
    # Setup
    setup_exchange(Π.lds, x)
    for ld in Π.lds # TODO parallelize
        Πxi = ld.Πi(x)
        local2global_trace(ld, Πxi, Π.s)
        axpy!(1, Π.s, Π.y)
    end
    return Π.y
end

struct Aop
    # Scattering operator
    S::GlobalScatteringOp
    # Exchange operator
    Π::ExchangeOp
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
struct DDM{T1,T2,S<:AbstractLocalData}
    # Local data
    lds::Array{S,1}
    # Iterative operator
    A::Aop
    # Lifting operator
    L::GlobalLiftingOp
    # Solution with physical source but 0 transmission BCs: u = Lx + F
    F::Array{T1,1} # vector of DOF
    # RHS of interface problem x = Ax + b
    b::Array{T2,1} # expressed in weak form <ik (Tij+Tji) Fi, G^t>
end
function DDM(lds, to)
    if length(lds) < 2 @warn "Number of domains < 2." end
    L = GlobalLiftingOp(lds)
    S = GlobalScatteringOp(lds)
    Π = ExchangeOp(lds)
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
    DDM(lds,A,L,F,b)
end
