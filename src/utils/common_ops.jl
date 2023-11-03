"""
    USED FOR COPYING THE INPUT DATA INTO THE PADDED ARRAYS FOR FFT
"""

# The two functions below accept a tuple of coordinates, which makes the code more succint.
# "coord_data" contains = [ IA_TLF, IA_BRB, SA_TLF, SA_BRB, SA_TLF_off, SA_BRB_off ].

function copy_inter_region!( pad_inter, input1, coord_data::NTuple{6,N} ) where {N}
    copy_inter_region!( pad_inter, input1, coord_data[1], coord_data[2] )
end

function copy_search_region!( pad_inter, input1, coord_data::NTuple{6,N} ) where {N}
    copy_search_region!( pad_inter, input1, coord_data[3], coord_data[4], coord_data[5] )
end

# 

function copy_inter_region!( pad_inter, input1, IR_TLF, IR_BRB )
    pad_inter .= 0.0; 
    inter_size = IR_BRB .- IR_TLF .+ 1; 
    pad_inter[ Base.OneTo.(inter_size)... ] .= input1[ UnitRange.(IR_TLF,IR_BRB)... ]
end

function copy_search_region!( pad_search, input2, SR_TLF, SR_BRB, SR_TLF_offset )

    pad_search .= 0.0
    search_size = SR_BRB .- SR_TLF .+ 1; 
    padsearch_coords = UnitRange.( 1 .+ SR_TLF_offset,  search_size .+ SR_TLF_offset );
    pad_search[ padsearch_coords... ] .= input2[ UnitRange.( SR_TLF, SR_BRB )... ]; 
end


"""
    USED MOSTELY FOR FILTERING INTERROGATION REGIONS THAT BELONG TO THE BACKGROUND. 
    GENERALLY, THOSE WITH A VERY LOW MEAN INTENSITY CORRESPOND TO BACKGROUND REGIONS.
"""
function skip_inter_region( input, IR_TLF, IR_BRB, pivparams::PIVParameters )

    if pivparams.threshold < 0
        return false
    else
        inter_view = view( input, UnitRange.( IR_TLF, IR_BRB )... ); 
        return pivparams.filtFun( inter_view ) < pivparams.threshold
    end
end


"""
    USED FOR FINDING THE MAXIUMUM PEAK IN THE CROSS-CORRELATION MATRIX. 
"""

function firstPeak( cmat::Array{<:AbstractFloat,N} ) where {N}
    maxindex = maxidx( cmat )
    maxcartx = linear2cartesian( maxindex, size(cmat) )
    return maxcartx, cmat[ maxindex ]
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

function linear2cartesian( lidx, input_size::Dims{2} )
	h = input_size[1]
    x = ceil( Int, lidx/h )
    y = lidx - (x-1)*h;
    return (y,x)
end

function linear2cartesian( lidx, input_size::Dims{3} )
	h = input_size[1]; 
    w = input_size[1];
    z = ceil( Int, lidx/(h*w) )
    x = ceil( Int, (lidx - (z-1)*h*w)/h )
    y = lidx - (x-1)*h - (z-1)*h*w;
    return (y,x,z)
end


"""
    GAUSSIAN SUBPIXEL APPROXIMATION.
"""

function gaussian_displacement( corr_mat, scale, pivparams::PIVParameters )

    # cross-correlation maximum peak coordinates 
    peak, maxval = firstPeak( corr_mat )

    # unpadded cross-correlation center is very simple, searchMargin .+ 1. 
    center = _smarg( pivparams, scale ) .+ 1; 

    displacement = gaussian_refinement( corr_mat, peak, maxval ) .- center

    return displacement
end

function gaussian_displacement( corr_mat, isize::Dims{N}, ssize::Dims{N} ) where {N}

    peak, maxval = firstPeak( corr_mat )
    center = div.( isize, 2 ) .+ div.( ssize, 2 )
    return gaussian_refinement( corr_mat, peak, maxval ) .- center
end

function gaussian_refinement( corr_mat::Array{T,N}, peak, maxval ) where {T,N}

    if all( peak .> 1 ) && all( peak .< size(corr_mat) )
        minval = minimum( corr_mat[ UnitRange.( peak .- ones(Int,N), peak .+ ones(Int,N) )... ] )
        gaussi = gaussian( corr_mat, peak, maxval, T(minval) )
        return peak .+ gaussi
    else
        return peak 
    end
end

# 3-point Gaussian subpixel 2D from the pixel neighbourhood around the max peak 
function gaussian( corr_mat::Array{T,2}, peak, maxval::T, minval::T=0 ) where {T}

    return gaussian_2D( log(1+corr_mat[peak[1]+1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1]-1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]-1]-minval), 
                        log(1+corr_mat[peak[1],peak[2]+1]-minval), 
                        log(1+maxval-minval) ); 
end

# 3-point Gaussian subpixel 3D from the voxel neighbourhood around the max peak
function gaussian( corr_mat::Array{T,3}, peak, maxval::T, minval::T=0 ) where {T<:AbstractFloat} 

    return gaussian_3D( log(1+corr_mat[peak[1]+1,peak[2],peak[3]]-minval), 
                        log(1+corr_mat[peak[1]-1,peak[2],peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]-1,peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]+1,peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2],peak[3]-1]-minval), 
                        log(1+corr_mat[peak[1],peak[2],peak[3]+1]-minval),
                        log(1+maxval-minval) ); 
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


"""
    USED FOR SMART PADDING FOR OPTIMAL FFT PERFORMANCE, TAKEN FROM SCIPY
"""

function root( base, goal, mul=1 )
	n = 0; 
	while ( mul * base ^ ( n + 1 ) < goal )
		n += 1
	end
	return n
end

needs_factorization( n ) = !any( [ n % f == 0  for f in (2,3,5,11) ] )

function good_size_real( n )

	( n <= 6 ) && ( return n; )
	
	bestfac = 2 * n
	for n5 in 0:root( 11, bestfac, 5 );  f5 = 5^n5
	  x = f5
		while ( x < n )
			x *= 2
		end
		while (true)
			if ( x < n )
				x *= 3
			elseif ( x > n )
				( x < bestfac ) && ( bestfac = x; )
				( x&1 == 1 ) && ( break; )
				x >>= 1
			else
				return n
			end
		end # while (true)	
	end
	return bestfac
end

function good_size_cmplx( n )

	( n <= 12 ) && ( return n; )
	
	bestfac = 2 * n; 
	for n11 in 0:root( 11, bestfac, 1 );             f11   = 11^n11
		for n117 in 0:root( 7, bestfac, f11 );       f117  = f11*7^n117
			for n1175 in 0:root( 5, bestfac, f117 ); f1175 = f117*5^n1175

				x = f1175; 
				while ( x < n )
					x *= 2
				end
				while (true)
					if ( x < n )
						x *= 3
					elseif ( x > n )
						( x < bestfac ) && ( bestfac = x; )
						( x&1 == 1 ) && ( break; )
						x >>= 1
					else
						return n
					end
				end # while (true)		
	end end end

	return bestfac
end