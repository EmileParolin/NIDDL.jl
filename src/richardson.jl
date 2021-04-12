mutable struct Jacobi_S <: Solver
    r::Real
    tol::Real
    maxit::Integer
    light_mode::Bool
    x::Array{Complex{Float64},1}
    Ax::Array{Complex{Float64},1}
end
function Jacobi_S(;tol=1.e-3, maxit=100, r=0.5, light_mode=true)
    @assert 0 <= r <= 1
    x = zeros(Complex{Float64},0)
    Ax = zeros(Complex{Float64},0)
    return Jacobi_S(r,tol,maxit,light_mode,x,Ax)
end


"""
The `resfunc` function should output an array of `Float64`
and it is assumed that the first element is the discrete l2 residual.
"""
function (s::Jacobi_S)(ddm::DDM; resfunc=(it)->zeros(Float64,1), to=missing)
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    A = ddm.A; b = ddm.b; L = ddm.L; F = ddm.F;
    if (length(s.x)==0 || size(s.x) != size(b)) s.x = 0*b end
    if (length(s.Ax)==0 || size(s.Ax) != size(b)) s.Ax = 0*b end
    # Residual (and other meaningful error)
    res = zeros(Float64,s.maxit+1,length(resfunc(0))) # for convergence plots
    res[1,:] = resfunc(0); @info "Iteration 0 at $(res[1,:])"
    # Starting iterations
    it = 1
    @timeit to "Iterations" begin
        while res[it,1]/res[1,1] > s.tol && it <= s.maxit
            # Matrix vector product
            t = @elapsed s.Ax = A(s.x)
            # L2 norm of the residual of the linear system (I-A)x = b (default)
            res[it+1,:] = resfunc(it)
            it%20 == 0 && @info "Iteration $(it) at $(res[it+1,:]) in $(t) seconds"
            # Relaxed Jacobi iteration
            # x = @. r*x + (1-r)*Ax + (1-r)*b
            lmul!(s.r,s.x)
            axpy!(1-s.r,s.Ax,s.x)
            axpy!(1-s.r,b,s.x)
            # Handling of divergence
            if isnan(res[it+1,1]) || res[it+1,1] > 1.e5
                @warn "Jacobi algorithm is diverging, stopping iterations."
                break
            end
            it += 1
        end
    end
    # Computing solution
    u = L(s.x) + F
    return u, s.x, res
end
