using FFTW

"""
    FFTW.jl does not implement in-place r2c, so we interface with FFTW's C-library 
    directly and follow  FFTW's documentation to implement in-place r2c and in-place
    c2r. Below are the functions to plan inplace r2c/c2r. 
""" 

# R2C 

function inplace_r2c_plan( pad_input::Array{Float64,N}, input_size, num_threads=Threads.nthreads() ) where {N}

    FFTW.set_num_threads(num_threads);

    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3.fftw_plan_dft_r2c(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{Cdouble}, pad_input::Ptr{ComplexF64}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

function inplace_r2c_plan( pad_input::Array{Float32,N}, input_size, num_threads=Threads.nthreads() ) where {N}

    FFTW.set_num_threads(num_threads);
    
    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3f.fftwf_plan_dft_r2c(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{Cfloat}, pad_input::Ptr{ComplexF32}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

# C2R

function inplace_c2r_plan( pad_input::Array{Float64,N}, input_size, num_threads=Threads.nthreads() ) where {N}

    FFTW.set_num_threads(num_threads); 
    
    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3.fftw_plan_dft_c2r(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{ComplexF64}, pad_input::Ptr{Cdouble}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

function inplace_c2r_plan( pad_input::Array{Float32,N}, input_size, num_threads=Threads.nthreads() ) where {N}
    
    FFTW.set_num_threads(num_threads);

    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3f.fftwf_plan_dft_c2r(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{ComplexF32}, pad_input::Ptr{Cfloat}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

"""
    Since we cannot rely on FFTW.jl rFFT! structs with default destructors, I also
    include functions to explicitely execute and destroy the r2c! and c2r! plans.
"""
 
function execute_plan!( p, pad_array::Array{Float64,N} ) where {N}
    @ccall FFTW.FFTW_jll.libfftw3.fftw_execute_dft_r2c(p::FFTW.PlanPtr, pad_array::Ptr{Cdouble}, pad_array::Ptr{ComplexF64})::Cvoid
end

function execute_plan!( p, pad_array::Array{Float32,N} ) where {N}
    @ccall FFTW.FFTW_jll.libfftw3f.fftwf_execute_dft_r2c(p::FFTW.PlanPtr, pad_array::Ptr{Cfloat}, pad_array::Ptr{ComplexF32})::Cvoid
end

function fftw_destroy_plan(p)
    @ccall FFTW.FFTW_jll.libfftw3.fftw_destroy_plan(p::FFTW.PlanPtr)::Cvoid
end

function fftw_cleanup()
    @ccall  FFTW.FFTW_jll.libfftw3.fftw_cleanup()::Cvoid
    if ( Threads.nthreads() > 1 )
        @ccall  FFTW.FFTW_jll.libfftw3.fftw_cleanup_threads()::Cvoid
    end
end

"""
    corr( F, G ) = IFFT( conj( FFT( F ) ) .* FFT( G ) )

    Implementing element-wise complex multiplication with conjugate 
    FFT(F), given that FFT(F) and FFT(G) are stores as real arrays 
    in r2c interleaved format.

    This funciton relies on the fact that length(pad_G) == even
"""

function corr_dot!( pad_G, pad_F )
    @inbounds @simd for idx in 1:2:length(pad_G)
        a = pad_G[ idx ]
        b = -1 * pad_G[idx+1]
        c = pad_F[ idx ]
        d = pad_F[idx+1]
        pad_G[ idx ] = a*c - b*d
        pad_G[idx+1] = a*d + c*b
    end
end

"""
    conv( F, G ) = IFFT( FFT( F ) ) .* FFT( G ) )

    Implementing element-wise complex multiplication given that FFT(F)
    and FFT(G) are stores as real arrays in r2c interleaved format.
"""

# ( a + bi )( c + di ) = ac + adi + cbi - bd = ac - bd + ( ad + cb )i

function conv_dot!( pad_G, pad_F )
    @inbounds @simd for idx in 1:2:length(pad_G)
        a = pad_G[ idx ]
        b = pad_G[idx+1]
        c = pad_F[ idx ]
        d = pad_F[idx+1]
        pad_G[ idx ] = a*c - b*d
        pad_G[idx+1] = a*d + c*b
    end
end

"""
    By definition the size of the cross-correlation matrix for each dimension is 
    "N + M - 1", which is the number of possible translations in each dimension. 
    In quickPIV we are only interested in fully overlapping translations, so we 
    are free to perform smaller FFTs of size "max( N, M )".

    On top to "max( N, M )", we need to add we add extra pixels (either 1 or 2) 
    to perform inplace real FFT transforms. These extra pixel(s) aren't actually
    part of the FFT transform, they simply  ensure that there is enough memory to
    store the results. 

    We can choose to add etra padding to the FFT transform with SciPy's algorithm
    to round up the dimensions of the cross-correlation matrix to the closest factor
    of 2,3,5 and 7. This increases the size of the FFT's, but it speeds up FFTs (for
    the most part). If not, we at least can add 1 pixel of padding to odd dimension
    to make them even, and slightly speed up FFT (for the most part).

"""
function compute_csize_and_paddings( size_F::Dims{N}, 
                                     size_G::Dims{N}; 
                                     unpadded=false, 
                                     good_pad=false, 
                                     odd_pad=false ) where {N}

  
    csize0   = unpadded ? max.( size_F, size_G ) : size_F .+ size_G .- 1; 
    csize1   = good_pad ? good_size_real.( csize0 ) : csize0 .+ isodd.( csize0 ) .* odd_pad;
    r2c_pad  = ( 1 + iseven( csize1[1] ), zeros(Int,N-1)... );
    corr_pad = csize1 .- csize0; 

    return csize1, r2c_pad, corr_pad
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