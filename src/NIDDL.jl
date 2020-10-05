module NIDDL

import Base: *

using LinearAlgebra
using SparseArrays
using SuiteSparse
using SharedArrays
using Distributed
using LinearMaps
using IterativeSolvers
using TimerOutputs

import LinearAlgebra: norm, size, transpose, mul!, ldiv!
import IterativeSolvers: gmres

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

abstract type AbstractProblem end
abstract type AbstractInputData end
abstract type AbstractGlobalData end
abstract type AbstractLocalData end

abstract type Solver end

indices_full_domain(gid::AbstractInputData) = error("Required by NIDDL")
indices_skeleton(gid::AbstractInputData) = error("Required by NIDDL")
indices_domain(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
indices_transmission_boundary(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
size_multi_trace(gid::AbstractInputData) = error("Required by NIDDL")
dof_weights(gid::AbstractInputData) = error("Required by NIDDL")
get_matrix(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
get_matrix_no_transmission_BC(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
get_rhs(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
get_transmission_matrix(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")
DtN(gid::AbstractInputData, pb::AbstractProblem) = error("Required by NIDDL")

include("mapping.jl")
include("ddm.jl")
include("onion.jl")
include("junctions.jl")
include("richardson.jl")
include("gmres.jl")

export
    AbstractProblem,
    AbstractInputData,
    AbstractGlobalData,
    AbstractLocalData,
    indices_full_domain, indices_skeleton, indices_domain, indices_transmission_boundary,
    size_multi_trace, dof_weights,
    get_matrix, get_rhs,
    DtN,

    LocalLiftingOp,
    LocalScatteringOp,
    LocalExchangeOp,
    DDM_Type,

    # mapping.jl
    Mapping,
    mapping_from_global_indices, matrix,
    size, transpose, mul!,

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
