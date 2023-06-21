using FFTW

"""
    FFTW.jl does not implement in-place r2c, so we interface with FFTW's C-library directly and 
    follow  FFTW's documentation to implement in-place r2c and in-place c2r. Below are the functions
    to plan inplace r2c/c2r. 
""" 

# R2C 

function inplace_r2c_plan( pad_input::Array{Float64,N}, input_size ) where {N}

    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3.fftw_plan_dft_r2c(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{Cdouble}, pad_input::Ptr{ComplexF64}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

function inplace_r2c_plan( pad_input::Array{Float32,N}, input_size ) where {N}
    
    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3f.fftwf_plan_dft_r2c(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{Cfloat}, pad_input::Ptr{ComplexF32}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

# C2R

function inplace_c2r_plan( pad_input::Array{Float64,N}, input_size ) where {N}
    
    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3.fftw_plan_dft_c2r(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{ComplexF64}, pad_input::Ptr{Cdouble}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

function inplace_c2r_plan( pad_input::Array{Float32,N}, input_size ) where {N}
    
    rank  = Cint( N );
    n     = Cint[ reverse( input_size )... ]; 
    flags = FFTW.ESTIMATE | FFTW.DESTROY_INPUT;
    plan  = @ccall FFTW.FFTW_jll.libfftw3f.fftwf_plan_dft_c2r(rank::Cint, n::Ptr{Cint}, pad_input::Ptr{ComplexF32}, pad_input::Ptr{Cfloat}, flags::Cuint)::FFTW.PlanPtr
    return plan
end

"""
    Since we cannot rely on FFTW.jl rFFT! structs with default destructors, I also include functions
    to explicitely execute and destroy the r2c! and c2r! plans.
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
end

"""
    corr( F, G ) = IFFT( conj( FFT( F ) ) .* FFT( G ) )

    Implementing element-wise complex multiplication given that FFT(F)
    and FFT(G) are stores as real arrays in r2c interleaved format.
"""

# ( a + bi )( c + di ) = ac + adi + cbi - bd = ac - bd + ( ad + cb )i

function corr_dot!( pad_G, pad_F )
    @inbounds @simd for idx in 1:2:length(pad_G)
        a = pad_G[ idx ]
        b = pad_G[idx+1]
        c = pad_F[ idx ]
        d = -1 * pad_F[idx+1]
        pad_G[ idx ] = a*c - b*d
        pad_G[idx+1] = a*d + c*b
    end
end