"""
Can be used to perform the mapping of data from the source to a target.
"""
struct Mapping
    lind_s::Vector{Int64}
    lind_t::Vector{Int64}
    dims::Tuple{Int64,Int64}
end


"""
Construct mapping from a global indices.
"""
function mapping_from_global_indices(gind_s::Vector{Int64}, gind_t::Vector{Int64})
    # Global indices of intersection
    gind_i = intersect(gind_s, gind_t)
    # Local indices
    lind_s = findall(i-> i in gind_i, gind_s)
    lind_t = findall(i-> i in gind_i, gind_t)
    # Size
    dims = sum.((length.(gind_t), length.(gind_s)))
    return Mapping(lind_s, lind_t, dims)
end


"""
Return the size as if it was a transformation matrix.
"""
size(mp::Mapping) = mp.dims
size(mp::Mapping, dim::Integer) = size(mp)[dim]


"""
Exchange the role of the source and target.
"""
function transpose(mp::Mapping)
    Mapping(mp.lind_t, mp.lind_s, reverse(mp.dims))
end


"""
Map the data from `us` (the source) to `ut` (the target).
"""
function *(mp::Mapping, us)
    ut = zeros(eltype(us), size(mp,1))
    mul!(ut, mp, us)
    return ut
end


"""
Map the data from `us` (the source) to `ut` (the target).
"""
function mul!(ut, mp::Mapping, us)
    @assert (length(ut), length(us)) == size(mp)
    setindex!(ut, getindex(us, mp.lind_s), mp.lind_t)
end


"""
Construct transformation matrix.
"""
function matrix(mp::Mapping)
    N = length(mp.lind_t)
    @assert N == length(mp.lind_s)
    sparse(mp.lind_t, mp.lind_s, ones(Bool, N), size(mp)...)
end
