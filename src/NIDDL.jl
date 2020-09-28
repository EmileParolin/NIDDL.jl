module NIDDL

using LinearAlgebra
using SparseArrays
using SuiteSparse
using SharedArrays
using Distributed
using LinearMaps
using IterativeSolvers
using TimerOutputs

import Base: +,-,*,/,zero
import LinearAlgebra: norm, ldiv!
import IterativeSolvers: gmres

using NIDDL_FEM

abstract type AbstractProblem end

abstract type BoundaryCondition end
abstract type PhysicalBC <: BoundaryCondition end
abstract type TransmissionBC <: BoundaryCondition end

"""
Local lifting operator: takes an incoming trace and solves the local
sub-problem.
"""
abstract type LocalLiftingOp end

"""
Local scattering operator: takes an incoming trace and computes an outgoing
trace.
"""
abstract type LocalScatteringOp end

"""Local exchange operator: takes a global trace and computes the local output
after exchange.

Note that there is a minus sign embedded in this exchange operator.
This sign comes from the fact that the implicit projector behind Π is a
projector on Neumann traces.
"""
abstract type LocalExchangeOp end

abstract type DDM_Type end

abstract type AbstractInputData end
abstract type AbstractGlobalData end
abstract type AbstractLocalData end

abstract type Solver end

dofdim() = error("Required by NIDDL")
get_matrix() = error("Required by NIDDL")
get_rhs() = error("Required by NIDDL")
transmission_boundary() = error("Required by NIDDL")
matrix() = error("Required by NIDDL")
DtN() = error("Required by NIDDL")

include("ddm.jl")
include("onion.jl")
include("junctions.jl")
include("richardson.jl")
include("gmres.jl")

export
    BoundaryCondition,
    PhysicalBC,
    TransmissionBC,
    AbstractProblem,

    LocalLiftingOp,
    LocalScatteringOp,
    LocalExchangeOp,
    DDM_Type,
    AbstractInputData,
    AbstractGlobalData,
    AbstractLocalData,

    dofdim,
    get_matrix,
    get_rhs,
    transmission_boundary,
    matrix,
    DtN,

    # ddm.jl
    init_global_solution,
    init_global_trace,
    global2local_trace,
    local2global_solution,
    local2global_trace,
    local_lifting_correction,
    GlobalLiftingOp,
    GlobalScatteringOp,
    GlobalExchangeOp,
    Aop,
    DDM,

    # onion.jl
    OnionDDM,
    OnionLocalData,
    OnionGlobalData,
    OnionLocalLiftingOp,
    OnionLocalScatteringOpImplicit,
    OnionLocalScatteringOpExplicit,
    OnionLocalExchangeOp,

    # junctions.jl
    JunctionsDDM,
    JunctionsLocalData,
    JunctionsGlobalData,
    JunctionsLocalLiftingOp,
    JunctionsLocalScatteringOp,
    JunctionsLocalExchangeOpImplicit,
    JunctionsLocalExchangeOpExplicit,

    # richardson.jl
    Jacobi_S, jacobi,

    # gmres.jl
    GMRES_S, gmres

end # module
