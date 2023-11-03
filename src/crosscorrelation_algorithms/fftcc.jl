""" 
  This function computes unnormalized cross-correlation in the frequency domain.
  It is the fastest  form of cross-correlation, but also the least robust. It is 
  expected to work accurately on traditional PIV datasets: bright dots moving 
  on a black background. In biology, you might get accurate results if you are
  analyzing the migration of fluorescently labelled nuclei, as they resemble 
  "bright dots on a black background".
"""

function displacement_from_crosscorrelation( ::FFTCC, scale, pivparams::PIVParameters, tmp_data )

  _FFTCC!( tmp_data..., scale, pivparams );

  displacement = gaussian_displacement( tmp_data[1], scale, pivparams )

  return displacement
end


"""
  In order to compute FFT cross-correlation we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/c2r FFT
    [3] FFT_plan_forward   : forward r2c FFT plan ( inplace )
    [4] FFT_plan_inverse   : inverse c2r FFT plan ( inplace )
    [5] r2c_pad            : we keep track of the padding used for r2c 
    [6] corr_pad           : we keep track of optimization corr padding
"""

function allocate_tmp_data( ::FFTCC, scale, pivparams::PIVParameters, precision=32 )

    return allocate_tmp_data( FFTCC(), _isize(pivparams,scale), _ssize(pivparams,scale),
                              precision=precision, unpadded=pivparams.unpadded, 
                              good_pad=pivparams.good_pad, odd_pad=pivparams.odd_pad
                             )
end

function allocate_tmp_data( ::FFTCC, 
                            isize::Dims{N},   # size of interrogation regions
                            ssize::Dims{N};   # size of search regions
                            precision=32,     # 32 or 64, for Float32 or Float64
                            unpadded=true,    # size padded arrays to "max( N, M )" instead of "N + M - 1".
                            good_pad=false,   # add extra padding to easy factorizatoin and speed up FFT.
                            odd_pad=true      # whether or not to round padded dimensions to even size
                          ) where {N}

  #=
    By definition the size of the cross-correlation matrix for each dimension is "N + M - 1", which
    is the number of possible translations in each dimension. In quickPIV we are only interested in
    fully overlapping translations, so we are free to perform smaller FFTs of size "max( N, M )".

    On top to "max( N, M )", we need to add we add extra pixels (either 1 or 2) to perform inplace
    real FFT transforms. These extra pixel(s) aren't actually part of the FFT transform, they simply 
    ensure that there is enough memory to store the results. 
  
    We can choose to add etra padding to the FFT transform with SciPy's algorithm to round up the
    dimensions of the cross-correlation matrix to the closest factor of 2,3,5 and 7. This increases  
    the size of the FFT's, but it speeds up FFTs (for the most part). If not, we at least can add 
    1 pixel of padding to odd dimension to make them even, and slightly speed up FFT (for the most
    part).
  =#
  
  csize0 = unpadded ? max.( isize, ssize ) : isize .+ ssize .- 1; 
  csize1 = good_pad ? good_size_real.( csize0 ) : csize0 .+ isodd.( csize0 ) .* odd_pad;

  r2c_pad  = ( 1 + iseven( csize1[1] ), zeros(Int,N-1)... );
  corr_pad = csize1 .- csize0; 

  pad_csize  = csize1 .+ r2c_pad;
  pad_ctype  = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( pad_ctype, pad_csize );
  pad_search = zeros( pad_ctype, pad_csize );
  r2c_plan   = inplace_r2c_plan( pad_inter , csize1 );
  c2r_plan   = inplace_c2r_plan( pad_search, csize1 );

  return ( pad_inter, pad_search, r2c_plan, c2r_plan )
end


"""
  Copying the interrogation and search regions into the padded arrays for the FFTs.
"""

function prepare_inputs!( ::FFTCC, F, G, coord_data, tmp_data )
  copy_inter_region!(  tmp_data[1], F, coord_data );  
  copy_search_region!( tmp_data[2], G, coord_data );   
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
  
  Check out FFTW's documentation for detailed information.
"""
function _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, scale, pivparams::PIVParameters )
  return _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, _isize(pivparams,scale), _ssize(pivparams,scale) ); 
end

function _FFTCC!( pad_F::Array{T,N}, # padded interrogation region
                  pad_G::Array{T,N}, # padded search region
                  r2c_plan,          # forward inplace real FFT plan
                  c2r_plan,          # backward inplace real FFT plan
                  size_F::Dims{N},   # interrogation size
                  size_G::Dims{N}    # search size
                ) where {T<:AbstractFloat,N}

  # FORWARD IN-PLACE TRANSFORMS OF PAD_G AND PAD_F. 
  execute_plan!( r2c_plan, pad_F )
  execute_plan!( r2c_plan, pad_G )

  # CONJ(PAD_F) .* PAD_G
  corr_dot!( pad_F, pad_G )

  # INVERSE IN-PLACE TRANSFORM OF PAD_G
  execute_plan!( c2r_plan, pad_F )

  # NORMALIZING THE RESULTS, OTHERWISE R2C/C2R SCALES BY THE NUMBER OF INPUTS 
  pad_F ./= T( prod( size_F .+ size_G ) )

  return nothing
end

"""
  Out-of-place implementation of FFTC for debugging. 
"""

function _FFTCC( F::Array{T,N}, G::Array{T,N}; 
                 precision=32, good_pad=false, unpadded=false ) where {T,N}

  tmp_data = allocate_tmp_data( FFTCC(), size(F), size(G), precision=precision, good_pad=good_pad, unpadded=unpadded )
  tmp_data[1][Base.OneTo.(size(F))...] .= F
  tmp_data[2][Base.OneTo.(size(G))...] .= G

  _FFTCC!( tmp_data..., size(F), size(G) ) 

  destroy_fftw_plans( FFTCC(), tmp_data )

  return tmp_data[1], tmp_data[2]
end


#=
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

function _FFTCC( F::Array{T,N}, G::Array{T,N}; precision=32, silent=true ) where {T,N}

    silent || println( "running non-piv FFT cross-correlation" ); 

    tmp_data = allocate_tmp_data( FFTCC(), size(F), size(G), precision )
    tmp_data[1][Base.OneTo.(size(F))...] .= F
    tmp_data[2][Base.OneTo.(size(G))...] .= G
    _FFTCC_nopiv!( tmp_data..., size(F), size(G) ) 
    destroy_fftw_plans( FFTCC(), tmp_data )

    return tmp_data[1], tmp_data[2]
end

function _FFTCC_nopiv!( pad_F::Array{T,N}, pad_G::Array{T,N}, r2c_plan, c2r_plan, pad1, pad2, size_G, size_F ) where {T<:AbstractFloat,N}

  execute_plan!( r2c_plan, pad_F )
  execute_plan!( r2c_plan, pad_G )
  corr_dot!( pad_F, pad_G )
  execute_plan!( c2r_plan, pad_F )
  pad_G ./= T( prod( size_F .+ size_G .- 1 ) )
  r2cpad = 1 + Int( iseven( size_F[1] + size_G[1] - 1 ) ); 
  circshift!( view( pad_G, 1:size(pad_G,1)-r2cpad, Base.OneTo.( size(pad_G)[2:end] )... ), 
              view( pad_F, 1:size(pad_F,1)-r2cpad, Base.OneTo.( size(pad_F)[2:end] )... ),
              size_F .- 1 )

  return nothing
end
=#