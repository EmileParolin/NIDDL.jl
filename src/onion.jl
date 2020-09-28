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
    MΩitoΩ::SparseMatrixCSC{Bool,Int64}
    # Mapping from local trace vector to global trace vector Σi → ΣMT
    MΣitoΣmt::SparseMatrixCSC{Bool,Int64}
    # Correction matrix used when computing lifting
    Ctbc_Σi::SparseMatrixCSC{Float64,Int64}
    Li::LocalLiftingOp
    Si::LocalScatteringOp
    Πi::LocalExchangeOp
    Fi::Array{Complex{Float64},1}
    bi::Array{Complex{Float64},1}
    # Timer
    to::TimerOutput
end


"""Local lifting operator: takes a trace and computes a local solution."""
struct OnionLocalLiftingOp{T} <: LocalLiftingOp where T
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
function OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},size(Kinv,1))
    Ui = zeros(Complex{Float64},size(Kinv,1))
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv, fi, Fi, Ui, ui)
end
function OnionLocalLiftingOp(MΣitoΩi, Kinv)
    fi = zeros(Complex{Float64},size(MΣitoΩi,1))
    Fi = zeros(Complex{Float64},0)
    Ui = zeros(Complex{Float64},0)
    ui = zeros(Complex{Float64},size(MΣitoΩi,1))
    OnionLocalLiftingOp(MΣitoΩi, 1, Kinv, fi, Fi, Ui, ui)
end
function (Li::OnionLocalLiftingOp{SparseMatrixCSC{Bool,Int64}})(Mxi)
    #Li.fi =-(Li.MΣitoΩi * Mxi)
    #Li.ui = Li.MKtoΩi * (Li.Kinv \ (transpose(Li.MKtoΩi) * Li.fi))
    mul!(Li.fi, Li.MΣitoΩi, Mxi)
    lmul!(-1, Li.fi)
    mul!(Li.Fi, transpose(Li.MKtoΩi), Li.fi)
    ldiv!(Li.Ui, Li.Kinv, Li.Fi)
    mul!(Li.ui, Li.MKtoΩi, Li.Ui)
    return Li.ui
end
function (Li::OnionLocalLiftingOp{Int64})(Mxi)
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
    Πi::SparseMatrixCSC{Complex{Float64},Int64}
    # Memory allocation
    Πxi::Vector{Complex{Float64}}
end
function OnionLocalExchangeOp(m, pbs, i, RΣ, RΣi, MΣitoΣmt, Cwsst)
    # Local part of projection operator
    Pi = hcat([RΣi * transpose(RΣ) * Cwsst * RΣ * transpose(restriction(m, transmission_boundary(pb), dofdim(pb)))
               for pb in pbs]...)
    # Local part of symmetry operator
    Πi = transpose(MΣitoΣmt) - 2 * Pi
    # Memory allocation
    Πxi = zeros(Complex{Float64}, size(Πi,1))
    return OnionLocalExchangeOp(Πi, Πxi)
end
(Πi::OnionLocalExchangeOp)(x) = mul!(Πi.Πxi, Πi.Πi, x)


function OnionLocalData(pbs::Vector{P}, gid::AbstractInputData, dd::OnionDDM,
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
            # Local matrix without TBC
            pb_noTBC = typeof(pb)(pb.medium, pb.Ω, filter(bc->typeof(bc)<:PhysicalBC, pb.BCs))
            Ktilde = get_matrix(gid.m,pb_noTBC,RΩi;to=to)
            MKtildetoΩi = sparse(I, number_of_elements(gid.m,pb.Ω,dofdim(pb)), size(Ktilde,1))
            KΣi = RΣi * transpose(RΩi) * MKtildetoΩi * Ktilde * transpose(MKtildetoΩi)
            # Local RHS
            if extension
                bi = - 2 * KΣi * Fi + 2 * transpose(MΣitoΩi) * MKtoΩi * transpose(MKtofi) * fi
            else
                bi = - 2 * KΣi * Fi + 2 * transpose(MΣitoΩi) * fi
            end
        end
        @timeit to "Lifting operator" begin
            if extension
                Li = OnionLocalLiftingOp(MΣitoΩi, MKtoΩi, Kinv)
            else
                Li = OnionLocalLiftingOp(MΣitoΩi, Kinv)
            end
        end
        @timeit to "Scattering operator" begin
            if dd.implicit
                Si = OnionLocalScatteringOpImplicit(Li, KΣi)
            else
                Σ = remove(union([bc.Γ for bc in pb.BCs
                                if typeof(bc) <: PhysicalBC]...),
                        boundary(pb.Ω))
                Λi = DtN(gid.m, pb, Σ)

                # Computation of transmission matrix
                Ti = spzeros(Float64,size(RΣi,1),size(RΣi,1))
                # Loop needed in case of more than one connected transmission boundary
                for bc in pb.BCs
                    if typeof(bc) <: TransmissionBC
                        RΣii = restriction(gid.m, bc.Γ, dofdim(pb))
                        Ti += RΣi * transpose(RΣii) * matrix(gid.m,pb,bc) * RΣii * transpose(RΣi)
                    end
                end
                Si = OnionLocalScatteringOpExplicit(Λi, Ti, Li)
            end
        end
        @timeit to "Exchange operator" begin
            Πi = OnionLocalExchangeOp(gid.m, pbs, i, gid.RΣ, RΣi, MΣitoΣmt, gid.Cwsst)
        end
    end
    return OnionLocalData(MΩitoΩ, MΣitoΣmt, Ctbc_Σi, Li, Si, Πi, Fi, bi, to)
end


localdata_type(dd::OnionDDM) = OnionLocalData
globaldata_type(dd::OnionDDM) = OnionGlobalData
