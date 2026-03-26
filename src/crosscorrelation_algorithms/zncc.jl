ZNCC_tmp{T,N} = Tuple{Array{T,N},Array{T,N},Array{T,N},Ptr{FFTW.fftw_plan_struct},Ptr{FFTW.fftw_plan_struct},NTuple{N,Int}};

"""
  This computes zero-normalized cross-correlation (ZNCC) efficiently by combining
  FFTCC in the frequency domain and integral arrays. Integral arrays are used
  to compute the sum of intensities in G at each translation. 

  ZNCC is equivalent to computing the normalize dot product between F and G at 
  each translation. 
"""

function crosscorrelation!( ::ZNCC, tmp_data::ZNCC_tmp, pivparams::PIVParameters )  
  _ZNCC!( tmp_data..., pivparams )  
  return nothing  
end

"""
  In order to compute ZNCC we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/cr2 FFT
    [3] Array{Float32/64,N}: integral array of ( search_region .^ 2 )
    [4] FFT_plan_forward : forward r2c FFT plan ( inplace )
    [5] FFT_plan_inverse : inverse c2r FFT plan ( inplace )
    [6] csize            : we keep track of the cross-correlation pad, after padding
"""

function allocate_tmp_data( ::ZNCC, pivparams::PIVParameters, precision::DataType=Float64 )
  return allocate_tmp_data( ZNCC(), _isize(pivparams), _ssize(pivparams), precision(1.0), pivparams.unpadded, pivparams.good_pad, pivparams.odd_pad )
end

function allocate_tmp_data( 
  ::ZNCC, 
  inter_size::Dims{N},  
  search_size::Dims{N},
  precision::T,
  unpadded=true,   
  good_pad=false,
  odd_pad=true  
)::ZNCC_tmp{T,N} where {
  T <: AbstractFloat,
  N
}
  csize, r2c_pad, corr_pad = compute_csize_and_paddings( inter_size, search_size, unpadded=unpadded, good_pad=good_pad, odd_pad=odd_pad )

  pad_csize  = csize .+ r2c_pad;
  pad_inter  = zeros( T,  pad_csize ); 
  pad_search = zeros( T,  pad_csize ); 
  int_search = zeros( T, search_size .+ 1 ); 
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );  
  c2r_plan   = inplace_c2r_plan( pad_search, csize );

  return ( pad_inter, pad_search, int_search, r2c_plan, c2r_plan, csize )  
end


"""
  Copying the interrogation and search regions into the padded arrays for the FFTs.
  In addition, we populate an integral array with the values^2 of the search region, 
  which will allow us to compute the L2 errors efficiently for each translation.
"""

function prepare_inputs!( 
  ::ZNCC, 
  tmp_data::ZNCC_tmp{T,N},
  input1::AbstractArray{<:Real,N}, 
  input2::AbstractArray{<:Real,N}, 
  coord_data, 
) where {
  T<:AbstractFloat,
  N
}
  copy_inter_region_minus_mean!(  tmp_data[1], input1, coord_data );          
  copy_search_region_minus_mean!( tmp_data[2], input2, coord_data );
  size2 = size(tmp_data[3]) .- 1; 
  integralArraySQ!( tmp_data[3], tmp_data[2], Tuple(ones(Int,N)), size2 );
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

function _ZNCC!( pad_F, pad_G, int2_G, r2c, c2r, csize, pivparams::PIVParameters )
  return _ZNCC!( pad_F, pad_G, int2_G, r2c, c2r, csize, _isize(pivparams), _ssize(pivparams) );
end

function _ZNCC!( 
  pad_F::Array{T,N}, 
  pad_G::Array{T,N}, 
  int2_G::Array{T,N}, 
  r2c_plan,
  c2r_plan, 
  csize,
  size_F::Dims{N}, 
  size_G::Dims{N} 
) where {
  T<:AbstractFloat,
  N
}
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
function _ZNCC( 
  input1::Array{T,N}, 
  input2::Array{T,N};
  precision=Float64, 
  good_pad=true, 
  unpadded=true, 
  odd_pad=false,
  fovp_crop=false
) where {
  T,
  N
}
  size_inp1, size_inp2 = size( input1 ), size( input2 ); 
  tmp_data = allocate_tmp_data( ZNCC(), size_inp1, size_inp2, precision(1.0), good_pad, unpadded, odd_pad )

  tmp_data[1][Base.OneTo.(size_inp1)...] .= input1
  tmp_data[2][Base.OneTo.(size_inp2)...] .= input2
  integralArraySQ!( tmp_data[3], tmp_data[2], ones(Int64,N), size_inp2 ); 
    
  _ZNCC!( tmp_data..., size_inp1, size_inp2 ); 

  destroy_fftw_plans( ZNCC(), tmp_data ); 

  out1 = tmp_data[1]
  if fovp_crop
    searchMargin = div.( size_inp2 .- size_inp1, 2 )
    fovp = UnitRange.( 1, searchMargin .+ 1 )
    out1 = tmp_data[1][fovp...]
  end

  return out1, tmp_data[2]
end