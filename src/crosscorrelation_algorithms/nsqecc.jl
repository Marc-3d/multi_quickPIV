NSQECC_tmp{T,N} = Tuple{Array{T,N},Array{T,N},Array{T,N},Ptr{FFTW.fftw_plan_struct},Ptr{FFTW.fftw_plan_struct},NTuple{N,Int}};

"""
  This computes normalized L2 error cross-correlation efficiently by combining
  FFTCC in the frequency domain and integral arrays to compute certain sums of
  the search region. 

  Remarkably, NSQECC looks for the translation that minimizes L2 errors between
  the interrogation and search regions, instead of the translation that maximizes
  their dot product. This leads in much more robust results for any datasets. 
  NSQECC is the default algorithm, and it is recommended for most biological 
  applications.
"""

function crosscorrelation!( ::NSQECC, tmp_data::NSQECC_tmp, pivparams::PIVParameters )  
  _NSQECC!( tmp_data..., pivparams )  
  return nothing
end

"""
  In order to compute NSQECC we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/cr2 FFT
    [3] Array{Float32/64,N}: integral array of ( search_region .^ 2 )
    [4] FFT_plan_forward   : forward r2c FFT plan ( inplace )
    [5] FFT_plan_inverse   : inverse c2r FFT plan ( inplace )
    [6] csize              : we keep track of the cross-correlation pad, after padding
"""

function allocate_tmp_data( ::NSQECC, pivparams::PIVParameters, precision::DataType=Float64 )
  return allocate_tmp_data( NSQECC(), _isize(pivparams), _ssize(pivparams), precision(1.0), pivparams.unpadded, pivparams.good_pad, pivparams.odd_pad )
end

function allocate_tmp_data( 
  ::NSQECC, 
  inter_size::Dims{N},  
  search_size::Dims{N},
  precision::T,
  unpadded=true,   
  good_pad=false,
  odd_pad=true  
)::NSQECC_tmp{T,N} where {
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
  ::NSQECC, 
  tmp_data::NSQECC_tmp{T,N},
  input1::AbstractArray{<:Real,N}, 
  input2::AbstractArray{<:Real,N}, 
  coord_data
) where {
  T,
  N
}
  copy_inter_region!(  tmp_data[1], input1, coord_data );          
  copy_search_region!( tmp_data[2], input2, coord_data );
  size2 = size(tmp_data[3]) .- 1; 
  integralArraySQ!( tmp_data[3], tmp_data[2], Tuple(ones(Int,N)), size2 );
  return nothing
end


"""
  Destroy FFTW plans from tmp_data
"""

function destroy_fftw_plans( ::NSQECC, tmp_data )
  fftw_destroy_plan( tmp_data[4] )
  fftw_destroy_plan( tmp_data[5] )
  fftw_cleanup()
end

"""
  At each translation, the sum of L2 differences between overlapping pixel expands to:

  (1)            sum( G - F )² = sum(G²) + sum(F²) - 2sum(G*F)

  where sum(G²) is constant, sum(F²) can be computed with an integral array, and sum(G*F)
  is computed by a standard cross-correlation. NSQECC is obtained by dividing (1) by
  sqrt( sum(G²) )*sqrt( sum(F²). Notice that these two quantities, sum(G²) and sum(F²), 
  have already been computed in (1), the numerator.

  NOTE: We only consider translations with full overlap between G and F, because it isn't
  obvious how to deal with partially overlaping translations when using the L2 similarity. 
"""

function _NSQECC!( pad_F, pad_G, int2_G, r2c, c2r, csize, pivparams::PIVParameters )
  return _NSQECC!( pad_F, pad_G, int2_G, r2c, c2r, csize, _isize(pivparams), _ssize(pivparams) );
end

function _NSQECC!( 
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

  # 3-. COMPUTING NSQECC IN-PLACE in PAD_F (only for fully-overlapping translations)
  fftcc2nsqecc!( pad_F, pad_G, int2_G, sumF2, size_F, size_G  )

  return nothing
end


"""
  Out-of-place implementation of FFTC for debugging and general cross-correlation
  applications. 

  This implementation does still assumes unpadded cross-correlation, so it does 
  NOT support circshifting. 
"""
function _NSQECC(
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
  size_inp1, size_inp2 = size( input1 ), size( input2 )
  tmp_data = allocate_tmp_data( NSQECC(), size_inp1, size_inp2, precision(1.0), unpadded, good_pad, odd_pad )
  tmp_data[1][Base.OneTo.(size_inp1)...] .= input1
  tmp_data[2][Base.OneTo.(size_inp2)...] .= input2
  integralArraySQ!( tmp_data[3], tmp_data[2], ones(Int64,N), size_inp2 ); 
  _NSQECC!( tmp_data..., size_inp1, size_inp2 ); 

  destroy_fftw_plans( NSQECC(), tmp_data ); 

  out1 = tmp_data[1]
  if fovp_crop
    searchMargin = div.( size_inp2 .- size_inp1, 2 )
    fovp = UnitRange.( 1, searchMargin .+ 1 )
    out1 = tmp_data[1][fovp...]
  end
  return out1, tmp_data[2]
end