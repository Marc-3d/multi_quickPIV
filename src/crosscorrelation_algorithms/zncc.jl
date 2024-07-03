"""
  This computes zero-normalized cross-correlation (ZNCC) efficiently by combining
  FFTCC in the frequency domain and integral arrays. Integral arrays are used
  to compute the sum of intensities in G at each translation. 

  ZNCC is equivalent to computing the normalize dot product between F and G at 
  each translation. 
"""

function crosscorrelation!( ::ZNCC, 
                            scale, 
                            pivparams::PIVParameters, 
                            tmp_data
                          )  

  _ZNCC!( tmp_data..., scale, pivparams )  
  return nothing
end

"""
  In order to compute ZNCC we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/cr2 FFT
    [3] Array{Float,N}   : integral array of ( search_region .^ 2 )
    [4] FFT_plan_forward : forward r2c FFT plan ( inplace )
    [5] FFT_plan_inverse : inverse c2r FFT plan ( inplace )
    [6] csize            : we keep track of the cross-correlation pad, after padding
"""

function allocate_tmp_data( ::ZNCC, scale, pivparams::PIVParameters, precision=32 )

  return allocate_tmp_data( ZNCC(), _isize(pivparams, scale), _ssize(pivparams, scale),
                            precision=precision, 
                            unpadded=pivparams.unpadded, 
                            good_pad=pivparams.good_pad, 
                            odd_pad=pivparams.odd_pad
                          )
end

function allocate_tmp_data( ::ZNCC, 
                            isize::Dims{N},  
                            ssize::Dims{N}; 
                            precision=32,
                            unpadded=true,   
                            good_pad=false,
                            odd_pad=true  
                          ) where {N}

  csize, r2c_pad, corr_pad = compute_csize_and_paddings( isize, ssize, 
                                                         unpadded=unpadded, 
                                                         good_pad=good_pad, 
                                                         odd_pad=odd_pad )

  pad_csize  = csize .+ r2c_pad;
  pad_ctype  = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( pad_ctype,  pad_csize ); 
  pad_search = zeros( pad_ctype,  pad_csize ); 
  int_search = zeros( pad_ctype, ssize .+ 1 ); 
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );  
  c2r_plan   = inplace_c2r_plan( pad_search, csize );

  return ( pad_inter, pad_search, int_search, r2c_plan, c2r_plan, csize )  
end


"""
  Copying the interrogation and search regions into the padded arrays for the FFTs.
  In addition, we populate an integral array with the values^2 of the search region, 
  which will allow us to compute the L2 errors efficiently for each translation.
"""

function prepare_inputs!( ::ZNCC, F::AbstractArray{T,N}, G::AbstractArray{T,N}, 
                                  coord_data, 
                                  tmp_data ) where {T,N}

  copy_inter_region_minus_mean!(  tmp_data[1], F, coord_data );          
  copy_search_region_minus_mean!( tmp_data[2], G, coord_data );

  size_G = size(tmp_data[3]) .- 1; 
  integralArraySQ!( tmp_data[3], tmp_data[2], Tuple(ones(Int64,N)), size_G );
end


"""
  Destroy FFTW plans from tmp_data
"""

function destroy_fftw_plans( ::ZNCC, tmp_data )
  fftw_destroy_plan( tmp_data[4] )
  fftw_destroy_plan( tmp_data[5] )
  fftw_cleanup()
end

"""
  At each translation, ZNCC computes:

  (1.1)   sum( ( F - mF ) .* ( G - mG ) ) / sqrt( sum( ( F - mF ).^ 2) * sqrt( ( G  - mG ) .^ 2 ) )
    |
    |  F_ = F - mF
    |  G_ = G - mG
    v
  (1.2)   sum( F_ .* G_ ) / sqrt( sum( F_ .^ 2) * sqrt( G_ .^ 2 ) )

  In other words, we need to: 

    1st: substract the means (mF and mG) from F and G. 
    2nd: the numerator in 1.2 is simply dot product between F_ and G_, which can be computed for
         each translation with FFTCC. 
    3rd: sum( F .^ 2 ) is a constant for all fully-overlapping translations.
    4rd: sum( G_ .^ 2 ) is a unique sum for each translation. Namely, it is the sum of the 
         pixels .^ 2 the rectangular region of G for each translation. These sums can be 
         computed in O(1) with an integral array. 

  NOTE: We only consider translations with full overlap between G and F, because it isn't
  obvious how to deal with partially overlaping translations when using the L2 similarity. 
"""

function _ZNCC!( pad_F, pad_G, int2_G, r2c, c2r, csize, scale, pivparams::PIVParameters )
  return _ZNCC!( pad_F, pad_G, int2_G, r2c, c2r, csize, _isize(pivparams,scale), _ssize(pivparams,scale) );
end

function _ZNCC!( pad_F::Array{T,N}, 
                 pad_G::Array{T,N}, 
                 int2_G::Array{T,N}, 
                 r2c_plan,
                 c2r_plan, 
                 csize,
                 size_F::Dims{N}, 
                 size_G::Dims{N} ) where {T<:AbstractFloat,N}

  # 1-. COMPUTING INTERROGATION REGION CONSTANTS, BEFORE ANY IN-PLACE FFT. 
  sumF2 = T(0.0)
  D = ( N == 2 ) ? 1 : size_F[3]; 
  for z in 1:D, x in 1:size_F[2], y in 1:size_F[1]
    @inbounds sumF2 += pad_F[y,x,z]^2
  end

  # 2-. COMPUTING REAL-DATA IN-PLACE CROSS-CORRELATION MATRIX, G ⋆ F, ON PAD_F.
  _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, csize ); 

  # 3-. COMPUTING ZNCC IN-PLACE in PAD_F (only for fully-overlapping translations)
  fftcc2zncc!( pad_F, pad_G, int2_G, sumF2, size_F, size_G  )

  return nothing
end


"""
  Out-of-place implementation of FFTC for debugging and general cross-correlation
  applications. 

  This implementation does still assumes unpadded cross-correlation, so it does 
  NOT support circshifting. 
"""
function ZNCC( F::Array{T,N}, G::Array{T,N};
                  precision=32, 
                  good_pad=false, 
                  unpadded=false, 
                  odd_pad=false ) where {T,N}

  tmp_data = allocate_tmp_data( ZNCC(), size(F), size(G), 
                                precision=precision, 
                                good_pad=good_pad, 
                                unpadded=unpadded, 
                                odd_pad=odd_pad )

  tmp_data[1][Base.OneTo.(size(F))...] .= F
  tmp_data[2][Base.OneTo.(size(G))...] .= G
  integralArraySQ!( tmp_data[3], tmp_data[2], ones(Int64,N), size(G) ); 
    
  _ZNCC!( tmp_data..., size(F), size(G) ); 

  destroy_fftw_plans( ZNCC(), tmp_data ); 

  return tmp_data[1], tmp_data[2]
end