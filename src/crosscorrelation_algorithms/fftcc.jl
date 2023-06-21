
""" 
  This function is the workhorse of PIV. It computes in-place cross-correlation of
  a pair of interrogation/search regions and computes a displacement vector. 

  This function corresponds to unnormalized cross-correlation computed with FFTs.
  It is the fastest form of cross-correlation, but also the least robust. It is 
  expected to work with nuclear labelled images, but not with non-segmentable data.
"""
function displacement_from_crosscorrelation( ::FFTCC, G, F, TLF, scale, pivparams, tmp_data )

  # SIZES OF INTERROGATOIN REGION (G) AND SEARCH REGION (F). 
  size_G = _isize( pivparams, scale ); 
  marg_F = _smarg( pivparams, scale );
  size_F = size_G .+ 2 .* marg_F;

  # COPYING INPUT DATA INTO THEIR RESPECTIVE PADDED ARRAYS
  prepare_inputs!( FFTCC(), G, F, TLF, size_G, marg_F, tmp_data )

  # COMPUTING NSQECC MATRIX INPLACE (ON PAD_G)
  _FFTCC!( tmp_data..., size_G, size_F );

  # FINDING MAXIMUM PEAK
  peak, maxval = firstPeak( tmp_data[2] );

  # COMPUTE DISPLACEMENT
  center = div.( size_F, 2 ) .+ div.( size_G, 2 ); 
  displacement = gaussian_displacement( tmp_data[2], peak, maxval, center )

  return displacement
end

"""
  In order to compute FFTCC we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/c2r FFT
    [3] FFT_plan_forward   : forward r2c FFT plan ( inplace )
    [4] FFT_plan_inverse   : inverse c2r FFT plan ( inplace )
"""
function allocate_tmp_data( ::FFTCC, scale, pivparams, precision=32 )
    return allocate_tmp_data( FFTCC(), _isize(pivparams,scale), _ssize(pivparams,scale), precision )
end

function allocate_tmp_data( ::FFTCC, isize::Dims{N}, ssize::Dims{N}, precision=32 ) where {N}

  # By definition the cross-correlation size for each dimension is "Ni + Mi - 1". In our case,
  # this computation always leads to odd numbers:
  #
  #    isize + ( isize + 2*smargin ) - 1   =   2*( isize + smargin ) - 1
  #
  # We add padding of 1 pixel/voxel to each dimension to make them even numbers. This speeds up 
  # the computation of FFTs (on average), by avoiding odd-prime numbers. In addition, we add 2
  # extra pixels/voxels on the first dimension to enable r2c/c2r inplace operation.

  csize      = isize .+ ssize .- 1 .+ 1;
  pad_csize  = csize .+ ( 2, zeros(Int,N-1)... );
  T          = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( T, pad_csize );
  pad_search = zeros( T, pad_csize );
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );
  c2r_plan   = inplace_c2r_plan( pad_search, csize );

  return ( pad_inter, pad_search, r2c_plan, c2r_plan )
end

"""
  Each interrogation area is defined by its top-left-corner (TLF) and the interrogation 
  size. Each search area is defined by the interrogation area (TLF + interSize) and the
  search margin around the interrogation area.  With this information we can copy the 
  interrogation/search regions into the padded arrays.
"""
function prepare_inputs!( ::FFTCC, G, F, TLF, size_G, marg_F, tmp_data )
  copy_inter_region!(  tmp_data[1], G, TLF, size_G );  
  copy_search_region!( tmp_data[2], F, TLF, size_G, marg_F );   
end

"""
  Destroy FFTW plans from tmp_data
"""
function destroy_fftw_plans( ::FFTCC, tmp_data )
  fftw_destroy_plan( tmp_data[3] )
  fftw_destroy_plan( tmp_data[4] )
  fftw_cleanup()
end

"""
  In-place computation of cross-correlation using FFTW's FFT functions for real-valued 
  data: r2c and c2r. These are supposed to be approximately twice as fast and requires
  half the memory than complex2complex FFT's. 
  
  Read FFTW's document for detailed information.
"""
function _FFTCC!( pad_G::Array{T,N}, pad_F::Array{T,N}, r2c_plan, c2r_plan, size_G, size_F ) where {T<:AbstractFloat,N}

  # forward in-place transforms of pad_G and pad_F. 
  execute_plan!( r2c_plan, pad_G )
  execute_plan!( r2c_plan, pad_F )

  # conj( pad_F ) .* pad_G.
  corr_dot!( pad_G, pad_F )

  # inverse in-place transform of pad_G.
  execute_plan!( c2r_plan, pad_G )

  # normalizing the results, otherwise r2c/c2r scales them by the number of inputs
  pad_G ./= T( prod( size_G .+ size_F ) )

  # circshifted results are stored in pad_F. 
  r2cpad = 2;
  shifts = size_F .- 1
  # TODO: check if using views leads to less allocations
  circshift!( view( pad_F, 1:size(pad_F,1)-r2cpad, Base.OneTo.( size(pad_F)[2:end] )... ), 
              view( pad_G, 1:size(pad_G,1)-r2cpad, Base.OneTo.( size(pad_G)[2:end] )... ),
              shifts )

  return nothing
end

#=
  Out-of-place FFTCC for debugging. 
  
  It assumes that size(G) .+ size(F) .- 1 is odd. It will fail if this is not the case.
=#
function _FFTCC_piv( G::Array{T,N}, F::Array{T,N}; precision=32 ) where {T,N}

  @assert all( iseven.( size(G) .+ size(F) ) ) "FFTCC will fail"

  println( "debugging FFT cross-correlation" ); 
  tmp_data = allocate_tmp_data( FFTCC(), size(G), size(F), precision )
  tmp_data[1][Base.OneTo.(size(G))...] .= G
  tmp_data[2][Base.OneTo.(size(F))...] .= F
  _FFTCC!( tmp_data..., size(G), size(F) ); 
  fftw_destroy_plan( tmp_data[3] )
  fftw_destroy_plan( tmp_data[4] )
  fftw_cleanup()
  return tmp_data[1], tmp_data[2]
end



"""
  Out-of-place implementation of FFT cross-correlation.
  ------------------------------------------------------------------------
  
  This function allows to apply FFT cross-correlation as a pattern matching
  operation of two inputs for general applications, not only PIV.

  In constrast with the PIV-specific implementation, the size of F is not
  size(G) .+ 2 .* search_marg, which is always an even number. Therefore, 
  we can't assume anything about the shape of the input data.
  
  This influences the construction of the r2c/c2r plans and the circshifting
  of the cross-correlation in the frequency domain. 
"""
function _FFTCC( G::Array{T,N}, F::Array{T,N}; precision=32 ) where {T,N}

    println( "running non-piv FFT cross-correlation" ); 

    tmp_data = allocate_tmp_data_nopiv( FFTCC(), size(G), size(F), precision=precision )
    
    tmp_data[1][Base.OneTo.(size(G))...] .= G
    tmp_data[2][Base.OneTo.(size(F))...] .= F
    
    _FFTCC_nopiv!( tmp_data..., size(G), size(F) ); 

    fftw_destroy_plan( tmp_data[3] )
    fftw_destroy_plan( tmp_data[4] )
    fftw_cleanup()

    return tmp_data[1], tmp_data[2]
end

function allocate_tmp_data_nopiv( ::FFTCC, isize::Dims{N}, ssize::Dims{N}; precision=32 ) where {N}
  T          = ( precision == 32 ) ? Float32 : Float64; 
  csize      = isize .+ ssize .- 1; 
  r2cpad     = 1 + iseven( csize[1] ); 
  pad_csize  = csize .+ ( r2cpad, zeros(Int,N-1)... );
  pad_inter  = zeros( T, pad_csize );
  pad_search = zeros( T, pad_csize );
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );
  c2r_plan   = inplace_c2r_plan( pad_search, csize );
  return ( pad_inter, pad_search, r2c_plan, c2r_plan ) 
end

function _FFTCC_nopiv!( pad_G::Array{T,N}, pad_F::Array{T,N}, r2c_plan, c2r_plan, size_G, size_F ) where {T<:AbstractFloat,N}

  execute_plan!( r2c_plan, pad_G )
  execute_plan!( r2c_plan, pad_F )
  corr_dot!( pad_G, pad_F )
  execute_plan!( c2r_plan, pad_G )
  pad_G ./= T( prod( size_G .+ size_F .- 1 ) )
  r2cpad = 1 + Int( iseven( size_G[1] + size_F[1] - 1 ) ); 
  circshift!( view( pad_F, 1:size(pad_F,1)-r2cpad, Base.OneTo.( size(pad_F)[2:end] )... ), 
              view( pad_G, 1:size(pad_G,1)-r2cpad, Base.OneTo.( size(pad_G)[2:end] )... ),
              size_F .- 1 )

  return nothing
end