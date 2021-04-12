mutable struct GMRES_S <: Solver
    tol::Real
    maxit::Integer
    restart::Integer
    light_mode::Bool
    x::Array{Complex{Float64},1}
    Ax::Array{Complex{Float64},1}
end
function GMRES_S(;tol=1.e-3, maxit=100, restart=20, light_mode=true)
    x = zeros(Complex{Float64},0)
    Ax = zeros(Complex{Float64},0)
    return GMRES_S(tol,maxit,restart,light_mode,x,Ax)
end


"""
The `resfunc` function should output an array of `Float64`
and it is assumed that the first element is the discrete l2 residual.

In GMRES implemented in IterativeSolvers, the computation of x at each
iteration is avoided (and Ax is not computed). x is only available at the end
and at restart. The output of the this function reflects this fact.
"""
function (s::GMRES_S)(ddm::DDM; resfunc=(it)->zeros(Float64,1), to=missing)
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    b = ddm.b; L = ddm.L; F = ddm.F;
    # Memory allocation
    if (length(s.x)==0 || size(s.x) != size(b)) s.x = 0*b end
    if (length(s.Ax)==0 || size(s.Ax) != size(b)) s.Ax = 0*b end
    # Creating LinearMap to feed in the GMRES solver (matrix free approach)
    function matvec(y, x)
        fill!(y,0)
        axpy!(1, x, y)
        axpy!(-1, ddm.A(x), y)
        return y
    end
    A = LinearMap{Complex{Float64}}(matvec, length(b);
                                    ismutating=true, issymmetric=false,
                                    ishermitian=false, isposdef=false)
    # Initialisation
    restart = min(s.restart,size(A,2))
    # GMRES iterator
    x = deepcopy(s.x)
    g = IterativeSolvers.gmres_iterable!(x, A, b;
                                         reltol=s.tol,
                                         maxiter=s.maxit,
                                         restart=restart,
                                         light_mode=s.light_mode)
    # Residual (or other types of error)
    res = zeros(Float64,s.maxit+1,length(resfunc(0))).+Inf # for convergence plots
    res[1,:] = resfunc(0); @info "Iteration 0 at $(res[1,:])"
    @timeit to "Iterations" for (it,resl2) in enumerate(g)
        # Computation of residual/error
        if !s.light_mode
            s.x = g.xbis
            s.Ax = ddm.A(s.x) # /!\ not always updated by GMRES
            res[it+1,:] = resfunc(it)
        end
        res[it+1,1] = resl2 # exact l2 residual
        it%20 == 0 && @info "Iteration $(it) at $(res[it+1,:])"
        if isnan(resl2) break end
    end
    # Computation of residual/error
    s.x = g.x
    s.Ax = ddm.A(s.x) # /!\ not always updated by GMRES
    @info "Converged at $(resfunc(Inf) ./ res[1,1]) (relative)"
    # Computing solution
    u = L(s.x) + F
    return u, s.x, res
end
