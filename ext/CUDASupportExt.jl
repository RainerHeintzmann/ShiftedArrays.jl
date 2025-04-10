module CUDASupportExt
using CUDA 
using Adapt
using ShiftedArrays
using Base 

get_base_arr(arr::CuArray) = arr
get_base_arr(arr::Array) = arr
function get_base_arr(arr::AbstractArray) 
    p = parent(arr)
    return (p === arr) ? arr : get_base_arr(parent(arr))
end

# define a number of Union types to not repeat all definitions for each type
AllShiftedTypeCu{N, CD} = Union{CircShiftedArray{<:Any,<:Any,<:CuArray{<:Any,N,CD}},
                                ShiftedArray{<:Any,<:Any,<:Any,<:CuArray{<:Any,N,CD}}}
AllShiftedTypeCuG{N, CD} = Union{AllShiftedTypeCu{N, CD}, CircShiftedArray{<:Any,<:Any,<:AllShiftedTypeCu{N,CD}},
                                 ShiftedArray{<:Any,<:Any,<:Any,<:AllShiftedTypeCu{N,CD}}}
AllSubArrayTypeCu{N, CD} = Union{SubArray{<:Any, <:Any, <:AllShiftedTypeCuG{N,CD}, <:Any, <:Any},
                                 Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCuG{N,CD}, <:Any},
                                 SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCuG{N,CD}, <:Any}, <:Any, <:Any}}
AllShiftedAndViewsCu{N, CD} = Union{AllShiftedTypeCuG{N, CD}, AllSubArrayTypeCu{N, CD}}

Adapt.adapt_structure(to, x::CircShiftedArray{T, N, S}) where {T, N, S} = CircShiftedArray(adapt(to, parent(x)), shifts(x));
Adapt.adapt_structure(to, x::ShiftedArray{T, V, N, S}) where {T, V, N, S} = ShiftedArray(adapt(to, parent(x)), shifts(x), default=ShiftedArrays.default(x));

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllShiftedTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllSubArrayTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

function Base.copy(s::AllShiftedAndViewsCu)
    res = similar(get_base_arr(s), eltype(s), size(s));
    res .= s
    return res
end

function Base.collect(x::AllShiftedAndViewsCu) 
    return copy(x) # stay on the GPU        
end

function Base.Array(x::AllShiftedAndViewsCu)
    return Array(copy(x)) # remove from GPU
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AbstractArray) 
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AbstractArray; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))), va...) 
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(y::AbstractArray, x::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...)     
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...) 
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.show(io::IO, mm::MIME"text/plain", cs::AllShiftedAndViewsCu) 
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end

# This version is needed to deal with range access of wrapped CuArrays.
# ShiftedVector(cu([1,2,3,4,5]))[2:3]
@inline function Base.getindex(s::AllShiftedTypeCu{N, CD}, x::Vararg{Union{AbstractRange, Int}, N}) where {N, CD}
    v = @view s[x...]
    res = similar(s.parent, eltype(s), size(v))
    res .= v
end

# This specializations are to ensure that true single element accesses generate an error, if allowscalar has not be specified.
@inline function Base.getindex(s::ShiftedArray{A,B,C, <:CuArray{<:Any,N,CD}}, x::Vararg{Int, N}) where {A,B,C, N,CD}
    invoke(ShiftedArrays.getindex, Tuple{ShiftedArray{A,B,C,<:AbstractArray}, ntuple((_)->Int, N)...}, s, x...)
end

@inline function Base.getindex(s::CircShiftedArray{A,B,<:CuArray{<:Any,N,CD}}, x::Vararg{Int, N}) where {A,B,N,CD}
    invoke(ShiftedArrays.getindex, Tuple{CircShiftedArray{A,B,<:AbstractArray}, ntuple((_)->Int, N)...}, s, x...)
end

end