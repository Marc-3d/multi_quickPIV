"""
    SIMD optimized (faster than Base.findmax) to find the max index of the
    cross-correlation matrices. Returns a linear index. 
"""

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

function maxval( a::AbstractArray{<:Real,N} ) where {N}
    max_idx = maxidx( a )
    return a[ max_idx ]; 
end


"""
    linaer2cartesian is used when finding coordinates of the maximum peak
    of the cross-correlation, and to find the vector field coordinates. 
"""

function linear2cartesian( lidx, input_size::Dims{2} )
	h = input_size[1]
    x = ceil( Int, lidx/h )
    y = lidx - (x-1)*h;
    return (y,x)
end

function linear2cartesian( lidx, input_size::Dims{3} )
	h = input_size[1]; 
    w = input_size[2];
    z = ceil( Int, lidx/(h*w) )
    x = ceil( Int, (lidx - (z-1)*h*w)/h )
    y = lidx - (x-1)*h - (z-1)*h*w;
    return (y,x,z)
end


"""
    USED FOR FINDING THE MAXIUMUM PEAK IN THE CROSS-CORRELATION MATRIX. 
"""

function firstPeak( cmat::Array{<:AbstractFloat,N} ) where {N}
    maxindex = maxidx( cmat )
    maxcartx = linear2cartesian( maxindex, size(cmat) )
    return maxcartx, cmat[ maxindex ]
end

# specialized function to find the maximum peak among only fully-overlapping translations
function find_max_translation( cmat::Array{T,2}, SM, SR_TLF_off, SR_BRB_off ) where {T}
    center    = SM .+  1; 
    trans_TLF = ones(N) .+ SR_TLF_off
    trans_BRB = ones(N) .* ( 2 .* SM .+ 1 ) .-  SR_BRB_off; 

    max_value = 0; 
    max_coord = ( 0, 0 ); 
    @inbounds for c in trans_TLF[2]:trans_BRB[2], 
                  r in trans_TLF[1]:trans_BRB[1]
        if cmat[ r, c ] > max_value
            max_value = cmat
            max_coord = ( r, c )
        end
    end
    return ( max_coord .- center, max_coord, max_value ); 
end

function find_max_translation( cmat::Array{T,3}, SM, SR_TLF_off, SR_BRB_off ) where {T}
    center    = SM .+  1; 
    trans_TLF = ( 1, 1 ).+ SR_TLF_off
    trans_BRB = ( 2 .* SM .+ 1 ) .-  SR_BRB_off; 

    max_value = 0; 
    max_coord = ( 0, 0, 0 ); 
    @inbounds for z in trans_TLF[3]:trans_BRB[3]
                  c in trans_TLF[2]:trans_BRB[2], 
                  r in trans_TLF[1]:trans_BRB[1]
        if cmat[ r, c, z ] > max_value
            max_value = cmat
            max_coord = ( r, c, z )
        end
    end
    return ( max_coord .- center, max_coord, max_value ); 
end


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