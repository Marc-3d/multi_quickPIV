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
    [5] csize              : we keep track of the cross-correlation pad, after padding
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

  csize1, r2c_pad, corr_pad = compute_csize_and_paddings( isize, ssize, 
                                                          unpadded=unpadded, 
                                                          good_pad=good_pad, 
                                                          odd_pad=odd_pad )

  pad_csize  = csize1 .+ r2c_pad;
  pad_ctype  = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( pad_ctype, pad_csize );
  pad_search = zeros( pad_ctype, pad_csize );
  r2c_plan   = inplace_r2c_plan( pad_inter , csize1 );
  c2r_plan   = inplace_c2r_plan( pad_search, csize1 );

  return ( pad_inter, pad_search, r2c_plan, c2r_plan, csize1 )
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

  Check out FFTW's documentation for detailed information about real FFT transforms.

  In addition, we assume "unpadded" cross-correlation for PIV purposes
"""

function _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, csize, scale, pivparams::PIVParameters )
  return _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, csize, _isize(pivparams,scale), _ssize(pivparams,scale) ); 
end

function _FFTCC!( pad_F::Array{T,N}, # padded interrogation region
                  pad_G::Array{T,N}, # padded search region
                  r2c_plan,          # forward inplace real FFT plan
                  c2r_plan,          # backward inplace real FFT plan
                  csize              # size of the cross-correlation matrix after padding
                ) where {T<:AbstractFloat,N}

  # FORWARD IN-PLACE TRANSFORMS OF PAD_G AND PAD_F. 
  execute_plan!( r2c_plan, pad_F )
  execute_plan!( r2c_plan, pad_G )

  # CONJ(PAD_F) .* PAD_G
  corr_dot!( pad_F, pad_G )

  # INVERSE IN-PLACE TRANSFORM OF PAD_G
  execute_plan!( c2r_plan, pad_F )

  # NORMALIZING THE RESULTS, OTHERWISE R2C/C2R SCALES BY THE NUMBER OF INPUTS 
  pad_F ./= T( prod( csize ) )

  return nothing
end


"""
  Out-of-place implementation of FFTC for debugging and general cross-correlation
  applications. 

  This implementation does not assume unpadded cross-correlation, so it supports
  circshifting. 
"""

function _FFTCC( F::Array{T,N}, G::Array{T,N}; 
                 precision=32, 
                 good_pad=false, 
                 unpadded=false, 
                 odd_pad=false, 
                 circshift=false ) where {T,N}

  tmp_data = allocate_tmp_data( FFTCC(), size(F), size(G), 
                                precision=precision, 
                                good_pad=good_pad, 
                                unpadded=unpadded, 
                                odd_pad=odd_pad )

  tmp_data[1][Base.OneTo.(size(F))...] .= F
  tmp_data[2][Base.OneTo.(size(G))...] .= G

  _FFTCC!( tmp_data... ) 

  destroy_fftw_plans( FFTCC(), tmp_data )

  if circshift
    csize   = tmp_data[end]
    padsize = size( tmp_data[1] ); 
    r2c_pad = padsize .- csize;
    Base.circshift!( view( tmp_data[2], UnitRange.( 1, padsize .- r2c_pad )... ), 
                     view( tmp_data[1], UnitRange.( 1, padsize .- r2c_pad )... ), 
                     size(F) .- 1 );     
    tmp_data[1] .= tmp_data[2]
  end

  return tmp_data[1], tmp_data[2]
end