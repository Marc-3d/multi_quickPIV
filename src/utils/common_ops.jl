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

function first_peak( cmat::Array{<:AbstractFloat,N} ) where {N}
    maxindex = maxidx( cmat )
    maxcartx = linear2cartesian( maxindex, size(cmat) )
    return maxcartx, cmat[ maxindex ]
end

function first_fullovp_peak( cmat::Array{T,N}, SM, SR_TLF_off, SR_BRB_off ) where {T,N}

    trans_TL = ones(Int64,N) .+ SR_TLF_off[1:N]
    trans_F  = ( N == 2 ) ? 1 : 1 + SR_TLF_off[3]
    trans_BR = ones(Int64,N) .+ 2 .* SM[1:N] .- SR_BRB_off[1:N]; 
    trans_B  = ( N == 2 ) ? 1 : 1 + 2 * SM[3] - SR_BRB_off[3]; 

    max_value = 0; 
    max_coord = ( 0, 0, 0 ); 
    @inbounds for z in trans_F:trans_B,
                  c in trans_TL[2]:trans_BR[2], 
                  r in trans_TL[1]:trans_BR[1]

        if cmat[ r, c, z ] > max_value
            max_value = cmat[ r, c, z ]
            max_coord = ( r, c, z )
        end
    end
    return ( max_coord[1:N], max_value ); 
end

function second_peak( cmat::Array{T,N}, peak1::Dims{N}, width=1 ) where {T,N}

	# Save the original values around the maximum peak before changing them to -Inf.
    ranges = UnitRange.( max.(1,peak1.-width), min.(size(cmat), peak1.+width) );
    OGvals = copy( cmat[ ranges... ] );
    cmat[ ranges... ] .= eltype(cmat)(-Inf);

	# Finding the second peak and copying back the original values.
    peak2, maxval2 = firstPeak( cmat );
    cmat[ ranges... ] .= OGvals;

    return peak2, maxval2
end

# SQUARE THE INPUT WHILE IN-PLACE CONSTRUCTION OF THE INTEGRAL ARRAY: used in zncc and nsqecc

function integralArraySQ!( int_array::AbstractArray{T,N}, 
                           array::AbstractArray{T,N}
                         ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, Tuple( ones(N) ), size( array ) )

    return nothing
end

function integralArraySQ!( int_array::AbstractArray{T,2}, 
                           array::AbstractArray{T,2}, 
                           TL=(1,1), 
                           inp_size=size(array) 
                         ) where {T<:AbstractFloat}
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        val2 = array[ ( TL .+ (r,c) .- 2 )... ]^2
        int_array[r,c] = val2 + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end

    return nothing
end

function integralArraySQ!( int_array::AbstractArray{T,3},
                           array::AbstractArray{T,3}, 
                           TLF=(1,1,1)::Dims{3}, 
                           inp_size=size(array) 
                         ) where {T<:AbstractFloat}
    h, w, d = inp_size;
    @inbounds for z in 1+1:d+1, c in 1+1:w+1
        tmp = 0.0; 
        for r in 1+1:h+1      
            val2 = array[ ( TLF .+ (r,c,z) .- 2 )... ]^2
            int_array[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
            tmp += val2; 
        end 
    end

    return nothing
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TL (TOP-LEFT) AND BR (BOTTOM-RIGHT) CORNERS

function integralArea( int_arr::Array{<:AbstractFloat,2}, TL, BR )
	TL   = TL .+ 1 .- 1;
	BR   = BR .+ 1;
	area = int_arr[BR[1],BR[2]] - int_arr[BR[1],TL[2]] - int_arr[TL[1],BR[2]] + int_arr[TL[1],TL[2]]
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TLF (TOP-LEFT-FRONT) AND BRB (BOTTOM-RIGHT-BACK) CORNERS

function integralArea( int_arr::Array{<:AbstractFloat,3}, TLF, BRB )
	TLF = TLF .+ 1 .- 1; 
	BRB = BRB .+ 1; 
	area  = int_arr[BRB[1],BRB[2],BRB[3]] - int_arr[TLF[1],TLF[2],TLF[3]]
	area -= int_arr[TLF[1],BRB[2],BRB[3]] + int_arr[BRB[1],TLF[2],BRB[3]] + int_arr[BRB[1],BRB[2],TLF[3]]
	area += int_arr[BRB[1],TLF[2],TLF[3]] + int_arr[TLF[1],BRB[2],TLF[3]] + int_arr[TLF[1],TLF[2],BRB[3]]
    return area
end

# COMPUTING SUMS FOR EACH TRANSLATION WITH INTEGRAL ARRAYS, used in znqecc and zncc

function add_numerator_nsqecc( pad_arr::AbstractArray{T1,2}, 
                               int_arr::AbstractArray{T1,2}, 
                               inter_size,
                               search_margin ) where {T1}

    # fully-overlapping translations
    pad_view = view( pad_arr, UnitRange.( 1, 1 .+ 2 .* search_margin )... ); 

    # 
    ih, iw = inter_size;
    mh, mw = search_margin;
    T , L  = 1:2*mh+1, 1:2*mw+1
    D , R  = 1+ih:2*mh+1+ih, 1+iw:2*mw+1+iw;

    pad_view .+= view( int_arr, D, R );
    pad_view .+= view( int_arr, T, L );
    pad_view .-= view( int_arr, T, R );
    pad_view .-= view( int_arr, D, L );
end

function add_numerator_nsqecc( pad_arr::AbstractArray{T1,3}, 
                               int_arr::AbstractArray{T1,3}, 
                               inter_size,
                               search_margin ) where {T1}

    # fully-overlapping translations
    pad_view = view( pad_arr, UnitRange.( 1, 1 .+ 2 .* search_margin )... ); 

    # 
    ih, iw, id = inter_size;
    mh, mw, md = search_margin;
    T , L , F  = 1:2*mh+1, 1:2*mw+1, 1:2*md+1
    D , R , B  = 1+ih:2*mh+1+ih, 1+iw:2*mw+1+iw, 1+id:2*md+1+id;

    pad_view .+= view( int_arr, D, R, B );
    pad_view .-= view( int_arr, T, L, F );
    pad_view .-= view( int_arr, T, R, B );
    pad_view .-= view( int_arr, D, L, B );
    pad_view .-= view( int_arr, D, R, F );
    pad_view .+= view( int_arr, D, L, F );
    pad_view .+= view( int_arr, T, R, F );
    pad_view .+= view( int_arr, T, L, B );
end