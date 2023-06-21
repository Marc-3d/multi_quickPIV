# FUNCTIONS TO COPY DATA INTO PADDED ARRAYS
# ------------------------------------------

# iTLF = interrogation region top-left-front corner
# iBRB = interrogation region bottom-right-back corner

function copy_inter_region!( pad_inter, inp1,  iTLF, isize )
    # Because inplace FFTs, we need to reset the pad_array 
    pad_inter .= 0.0; 
    iBRB = iTLF .+ isize .- 1;
    @inbounds pad_inter[ Base.OneTo.(isize)... ] .= inp1[ UnitRange.(iTLF,iBRB)... ]
end

function copy_search_region!( pad_search, inp2, iTLF, isize, smarg )

    iBRB = iTLF .+ isize .- 1; 

    # For search regions, both TLF and BRB may extends outside of the input array. 
    # The TLF outofbounds extension of the search region needs to be accounted  
    # when copying the input array into the padded search array. 
    outofbounds_sTLF = abs.( min.( iTLF .- smarg .- 1, 0 ) ); 
    search_coords    = UnitRange.( max.( 1, iTLF .- smarg ), min.( size(inp2), iBRB .+ smarg ) );
    padsearch_coords = UnitRange.( outofbounds_sTLF .+ 1, outofbounds_sTLF .+ length.( search_coords ) );

    pad_search .= 0.0
    pad_search[ padsearch_coords... ] .= inp2[ search_coords... ]; 
end

# FILTERING BACKGROUND IN GRAYSCALE IMAGES
# ----------------------------------------

function skip_inter_region( input, iTLF, scale, pivparams )
    if pivparams.threshold < 0
        return false
    end
    isize      = _isize(pivparams,scale)
    inter_view = view( input, UnitRange.( iTLF, iTLF .+ isize .- 1 )... ); 
    return pivparams.filtFun( inter_view ) < pivparams.threshold
end

# FINDING MAX PEAK IN THE CROSS-CORRELATION MATRIX
# -------------------------------------------------

# optimized to use SIMD, faster than Base.maximum
function maxval( a::AbstractArray{<:Real,N} ) where {N}

	len = length(a)
	if ( len < 4 )
        return maximum( a ); 
    else
        m1 = a[1]
        m2 = a[2]
        m3 = a[3]
        m4 = a[4]
        simdstart = len%4 + 1
        @inbounds @simd for i in simdstart:4:len
            m1 = ( a[ i ] > m1 ) ? a[ i ] : m1
            m2 = ( a[i+1] > m2 ) ? a[i+1] : m2
            m3 = ( a[i+2] > m3 ) ? a[i+2] : m3
            m4 = ( a[i+3] > m4 ) ? a[i+3] : m4
        end
        return maximum( ( m1, m2, m3, m4 ) )
    end
end

# optimized to use SIMD, faster than Base.findmax
function maxidx( a::AbstractArray{<:Real,N} ) where {N}

	len = length(a)
	if ( len < 4 )
        return findmax( a )[2];
    else
        maxidx1 = 1
        maxidx2 = 2
        maxidx3 = 3
        maxidx4 = 4
        simdstart = len%4 + 1
        @inbounds @simd for i in simdstart:4:len
            maxidx1 = ( a[ i ] > a[maxidx1] ) ? i   : maxidx1
            maxidx2 = ( a[i+1] > a[maxidx2] ) ? i+1 : maxidx2
            maxidx3 = ( a[i+2] > a[maxidx3] ) ? i+2 : maxidx3
            maxidx4 = ( a[i+3] > a[maxidx4] ) ? i+3 : maxidx4
        end
        # Finding the max value among the maximum indices computed in parallel
        val, idx = findmax( ( a[maxidx1], a[maxidx2], a[maxidx3], a[maxidx4] ) );
        maxindex = ( maxidx1, maxidx2, maxidx3, maxidx4 )[idx]
        return maxindex
    end
end

# maxidx + linear2cartesian
function firstPeak( cmat::Array{<:AbstractFloat,N} ) where {N}
    maxindex = maxidx( cmat )
    maxcartx = linear2cartesian( maxindex, cmat )
    return maxcartx, cmat[ maxindex ]
end

# masking 1st peak + maxidx + linear2cartesian 
function secondPeak( cmat::Array{T,N}, peak1::Dims{N}, width=1 ) where {T,N}

	# Save the original values around the maximum peak before changing them to -Inf.
    ranges = UnitRange.( max.(1,peak1.-width), min.(size(cmat), peak1.+width) );
    OGvals = copy( cmat[ ranges... ] );
    cmat[ ranges... ] .= eltype(cmat)(-Inf);

	# Finding the second peak and copying back the original values.
    peak2, maxval2 = firstPeak( cmat );
    cmat[ ranges... ] .= OGvals;

    return peak2, maxval2
end

# suitable for 2D and 3D arrays
function linear2cartesian( lidx, cmat::Array{T,N} ) where {T,N}
	h, w, d = size(cmat,1), size(cmat,2), size(cmat,3)
    z = ceil( Int, lidx/(h*w) )
    x = ceil( Int, (lidx - (z-1)*h*w)/h )
    y = lidx - (x-1)*h - (z-1)*h*w;
    return (y,x,z)[1:N]
end

# suitable for any dimensional array, but slower (15ns vs 30ns)
function linear2cartesian_2( lidx, arr::Array{T,N} ) where {T,N}
    linear2cartesian_2( lidx, strides(arr) ); 
end

function linear2cartesian_2( lidx, strides::Dims{N} ) where {N}
    cartx = zeros( Int, N ); 
    cartx[end] = div( (lidx - 1), strides[end] ) + 1;
    for i in N-1:-1:1
        lidx    -= strides[i+1]*( cartx[i+1] - 1 )
        cartx[i] = floor( Int, ( lidx - 1 )/strides[i] ) + 1;  
    end
    return Tuple( cartx )
end

# for fun
function size2strides( array_size::Dims{N} ) where {N}
    strides = ones( Int, N ) 
    for i in 2:N
        strides[i] *= strides[i-1] * array_size[i-1]
    end
    return Tuple( strides )
end

# GAUSSIAN SUBPIXEL WORKS FINE IN ALL PRACTICAL CASES.
# ----------------------------------------------------

function gaussian_displacement( corr_mat, scale, pivparams )
    # max cross-correlation peak
    peak, maxval = firstPeak( corr_mat )
    # cross-correlation center
    center = div.( _isize( pivparams, scale ), 2 ) .+ div.( _ssize( pivparams, scale ), 2 )
    # applying gaussian refinement
    return gaussian_refinement( corr_mat, peak, maxval ) .- center
end

function gaussian_refinement( corr_mat::Array{T,N}, peak, maxval ) where {T,N}
    if all( peak .> 1 ) && all( peak .< size(corr_mat) )
        minval = minimum( corr_mat[ UnitRange.( peak .- ones(Int,N), peak .+ ones(Int,N) )... ] )
        return peak .+ gaussian( corr_mat, peak, maxval, T(minval) )
    else
        return peak 
    end
end

# 3-point Gaussian subpixel 2D from the pixel neighbourhood around the max peak 
function gaussian( corr_mat::Array{T,2}, peak, maxval::T, minval::T=0 ) where {T}
    return gaussian_2D( log(corr_mat[peak[1]+1,peak[2]]-minval), 
                        log(corr_mat[peak[1]-1,peak[2]]-minval), 
                        log(corr_mat[peak[1],peak[2]-1]-minval), 
                        log(corr_mat[peak[1],peak[2]+1]-minval), 
                        log(maxval-minval) ); 
end

# 3-point Gaussian subpixel 3D from the voxel neighbourhood around the max peak
function gaussian( corr_mat::Array{T,3}, peak, maxval::T, minval::T=0 ) where {T<:AbstractFloat} 
    return gaussian_3D( log(corr_mat[peak[1]+1,peak[2],peak[3]]-minval), 
                        log(corr_mat[peak[1]-1,peak[2],peak[3]]-minval), 
                        log(corr_mat[peak[1],peak[2]-1,peak[3]]-minval), 
                        log(corr_mat[peak[1],peak[2]+1,peak[3]]-minval), 
                        log(corr_mat[peak[1],peak[2],peak[3]-1]-minval), 
                        log(corr_mat[peak[1],peak[2],peak[3]+1]-minval),
                        log(maxval-minval) ); 
end

function gaussian_2D( up::T, down::T, left::T, right::T, mx::T ) where {T<:AbstractFloat}
    return [ ( up  - down )/( 2* up  - 4*mx + 2*down  ), 
             (left - right)/( 2*left - 4*mx + 2*right ) ];
end

function gaussian_3D( up::T, down::T, left::T, right::T, front::T, back::T, mx::T ) where {T}
    return [ (  up  - down )/( 2*  up  - 4*mx + 2*down  ),
             ( left - right)/( 2* left - 4*mx + 2*right ),
             (front - back )/( 2*front - 4*mx + 2*back  ) ];
end