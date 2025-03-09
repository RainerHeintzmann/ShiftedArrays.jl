"""
    MutableShiftedArray(parent::AbstractArray, shifts, default)

Custom `AbstractArray` object to store an `AbstractArray` `parent` shifted by `shifts` steps
(where `shifts` is a `Tuple` with one `shift` value per dimension of `parent`).
As opposed to `ShiftedArray`, this object is mutable and mutation operations in the padded ranges are ignored.
For `s::MutableShiftedArray`, `s[i...] == s.parent[map(-, i, s.shifts)...]` if `map(-, i, s.shifts)`
is a valid index for `s.parent`, and `s.v[i, ...] == default` otherwise.
Use `copy` to collect the values of a `MutableShiftedArray` into a normal `Array`.
The recommended constructor is `MutableShiftedArray(parent, shifts; default = missing)`.

!!! note
    If `parent` is itself a `MutableShiftedArray` with a compatible default value,
    the constructor does not nest `MutableShiftedArray` objects but rather combines
    the shifts additively.

# Examples

```jldoctest shiftedarray
julia> v = [1, 3, 5, 4];

julia> s = MutableShiftedArray(v, (1,))
4-element MutableShiftedArray{Int64, Missing, Vector{Int64}}:
  missing
 1
 3
 5

julia> v = [1, 3, 5, 4];

julia> s = MutableShiftedArray(v, (1,))
4-element MutableShiftedArray{Int64, Missing, Vector{Int64}}:
  missing
 1
 3
 5

julia> copy(s)
4-element Vector{Union{Missing, Int64}}:
  missing
 1
 3
 5

julia> v = reshape(1:16, 4, 4);

julia> s = MutableShiftedArray(v, (0, 2))
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
 missing  missing  1  5
 missing  missing  2  6
 missing  missing  3  7
 missing  missing  4  8

julia> shifts(s)
(0, 2)
```
"""
struct MutableShiftedArray{T, M, N, S<:AbstractArray} <: AbstractArray{Union{T, M}, N}
    parent::S
    shifts::NTuple{N, Int}
    default::M
end

# low-level private constructor to handle type parameters
function mutableshiftedarray(v::AbstractArray{T, N}, shifts, default::M) where {T, N, M}
    return MutableShiftedArray{T, M, N, typeof(v)}(v, padded_tuple(v, shifts), default)
end

function MutableShiftedArray(v::AbstractArray, n = (); default = ShiftedArrays.default(v))
    return if (v isa MutableShiftedArray || v isa ShiftedArray) && default === ShiftedArrays.default(v)
        shifts = map(+, ShiftedArrays.shifts(v), padded_tuple(v, n))
        mutableshiftedarray(parent(v), shifts, default)
    else
        mutableshiftedarray(v, n, default)
    end
end

"""
    MutableShiftedVector{T, S<:AbstractArray}

Shorthand for `MutableShiftedArray{T, 1, S}`.
"""
const MutableShiftedVector{T, M, S<:AbstractArray} = MutableShiftedArray{T, M, 1, S}

function MutableShiftedVector(v::AbstractVector, n = (); default = ShiftedArrays.default(v))
    return MutableShiftedArray(v, n; default = default)
end

size(s::MutableShiftedArray) = size(parent(s))
axes(s::MutableShiftedArray) = axes(parent(s))

# Computing a shifted index (subtracting the offset)
offset(offsets::NTuple{N,Int}, inds::NTuple{N,Int}) where {N} = map(-, inds, offsets)

@inline function getindex(s::MutableShiftedArray{<:Any, <:Any, N}, x::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(s, x...)
    v, i = parent(s), offset(shifts(s), x)
    return if checkbounds(Bool, v, i...)
        @inbounds v[i...]
    else
        default(s)
    end
end

@inline function setindex!(s::MutableShiftedArray{<:Any, <:Any, N}, el, x::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(s, x...)
    v, i = parent(s), offset(shifts(s), x)
    if checkbounds(Bool, v, i...)
        @inbounds v[i...] = el
    end
    return s
end

parent(s::MutableShiftedArray) = s.parent

"""
    shifts(s::ShiftedArray)

Return amount by which `s` is shifted compared to `parent(s)`.
"""
shifts(s::MutableShiftedArray) = s.shifts

"""
    default(s::MutableShiftedArray)

Return default value.
"""
default(s::MutableShiftedArray) = s.default

default(::AbstractArray) = missing