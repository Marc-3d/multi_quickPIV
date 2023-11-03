"""
  This computes normalized L2 error cross-correlation efficiently by combining
  FFTCC in the frequency domain and integral arrays to compute certain sums of
  the values in the search region. 

  Remarkably, NSQECC looks for the translation that minimizes L2 errors between
  the interrogation and search regions, instead of the translation that maximizes
  their dot product. This leads in much more robust results for any datasets. NSQECC
  is the default algorithm, and it is recommended for most biological applicatoins.
"""

function displacement_from_crosscorrelation( ::NSQECC, scale, pivparams::PIVParameters, tmp_data )  

  _NSQECC!( tmp_data..., scale, pivparams )  

  displacement = gaussian_displacement( tmp_data[1], scale, pivparams )
  
  return displacement
end


"""
  In order to compute NSQECC we need: 
    [1] Array{Float32/64,N}: padded interr_region for r2c/c2r FFT
    [2] Array{Float32/64,N}: padded search_region for r2c/cr2 FFT
    [3] Array{Float,N}   : integral array of ( search_region .^ 2 )
    [4] FFT_plan_forward : forward r2c FFT plan ( inplace )
    [5] FFT_plan_inverse : inverse c2r FFT plan ( inplace )
"""
function allocate_tmp_data( ::NSQECC, scale, pivparams::PIVParameters, precision=32 )

  return allocate_tmp_data( NSQECC(), _isize(pivparams, scale), _ssize(pivparams, scale),
                            precision=precision, unpadded=pivparams.unpadded, 
                            good_pad=pivparams.good_pad, odd_pad=pivparams.odd_pad
                          )
end

function allocate_tmp_data( ::NSQECC, 
                            isize::Dims{N},  
                            ssize::Dims{N}; 
                            precision=32,
                            unpadded=true,   
                            good_pad=false,
                            odd_pad=true  
                          ) where {N}

  # csize1 = final size of FFT's in each dimension
  csize0   = unpadded ? max.( isize, ssize ) : isize .+ ssize .- 1; 
  csize1   = good_pad ? good_size_real.( csize0 ) : csize0 .+ isodd.( csize0 ) .* odd_pad;
  corr_pad = csize1 .- csize0; 

  # r2c_pad doesn't affect the r2c_plans, but it adds memory to the input arrays
  r2c_pad    = ( 1 + iseven( csize1[1] ), zeros(Int,N-1)... );
  pad_csize  = csize1 .+ r2c_pad;
  pad_ctype  = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( pad_ctype,  pad_csize ); 
  pad_search = zeros( pad_ctype,  pad_csize ); 
  int_search = zeros( pad_ctype, ssize .+ 1 ); 
  r2c_plan   = inplace_r2c_plan( pad_inter , csize1 );  
  c2r_plan   = inplace_c2r_plan( pad_search, csize1 );

  return ( pad_inter, pad_search, int_search, r2c_plan, c2r_plan )  
end


"""
  Copying the interrogation and search regions into the padded arrays for the FFTs.
  In addition, we populate an integral array with the values^2 of the search region, 
  which will allow us to compute the L2 errors efficiently for each translation.
"""
function prepare_inputs!( ::NSQECC, F, G::Array{T,N}, coord_data, tmp_data ) where {T,N}

  copy_inter_region!( tmp_data[1], F, coord_data );          
  copy_search_region!( tmp_data[2], G, coord_data );

  size_G = size(tmp_data[3]) .- 1; # search region size
  integralArraySQ!( tmp_data[3], tmp_data[2], ones(Int64,N), size_G );
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
function _NSQECC!( pad_G, pad_F, int2_F, r2c, c2r, scale, pivparams::PIVParameters )
  return _NSQECC!( pad_G, pad_F, int2_F, r2c, c2r, _isize(pivparams,scale), _ssize(pivparams,scale) );
end

function _NSQECC!( pad_F::Array{T,2}, 
                   pad_G::Array{T,2}, 
                   int2_G::Array{T,2}, 
                   r2c_plan,
                   c2r_plan, 
                   size_F::Dims{2}, 
                   size_G::Dims{2} ) where {T<:AbstractFloat}

  # COMPUTING INTERROGATION REGION CONSTANTS, BEFORE ANY IN-PLACE FFT. 
  sumF2 = T(0.0)
  for x in 1:size_F[2], y in 1:size_F[1]
    @inbounds sumF2 += pad_F[y,x]^2
  end
  sqrtsumF2 = sqrt( sumF2 )

  # COMPUTING REAL-DATA IN-PLACE CROSS-CORRELATION MATRIX, G ⋆ F, ON PAD_F.
  _FFTCC!( pad_F, pad_G, r2c_plan, c2r_plan, size_F, size_G ); 

  # COMPUTING NSQECC IN-PLACE (only for fully-overlapping translations)
  SM = div.( size_G .- size_F, 2 );
  for x in 1:2*SM[2], y in 1:2*SM[1]
      sumG2 = abs( integralArea( int2_G, (y,x), size_F .+ (y,x) ) )
      sumFG = pad_F[ y, x ]
      num   = sumF2 + sumG2 - 2*sumFG
      den   = sqrtsumF2 * sqrt(sumG2)
      L2    = num / den; 
      pad_F[ y, x ] = 1 / ( 1 + L2 )      
  end

  return nothing
end

""" 3D! """
function _NSQECC!( pad_F::Array{T,3}, pad_G::Array{T,3}, int2_G::Array{T,3}, r2c, c2r, size_F::Dims{3}, size_G::Dims{3} ) where {T<:AbstractFloat}

  sumF2 = 0.0
  @inbounds for z in 1:size_F[3], x in 1:size_F[2], y in 1:size_F[1]
    sumF2 += pad_F[y,x,z]^2
  end
  sqrtsumF2 = sqrt( sumF2 ) 

  _FFTCC!( pad_F, pad_G, r2c, c2r, size_F, size_G ); 

  SM = div.( size_G .- size_F, 2 );
  for z in 1:2*SM[3], x in 1:2*SM[2], y in 1:2*SM[1]
      sumG2 = abs( integralArea( int2_G, (y,x,z), size_F .+ (y,x,z) ) )
      sumFG = pad_F[ y, x, z ]
      num   = sumF2 + sumG2 - 2*sumFG
      den   = sqrtsumF2 * sqrt(sumG2)
      L2    = num / den; 
      pad_F[ y, x, z ] = 1 / ( 1 + L2 )      
  end

  return nothing
end

#=
  OUT-OF-PLACE NSQECC FOR DEBUGGING. 
  
  IT ASSUMES THAT size(G) .+ size(F) .- 1 IS ODD. IT WILL FAIL IF THIS IS NOT THE CASE.
=#
function _NSQECC_piv( G::Array{T,N}, F::Array{T,N}; precision=32, silent=true ) where {T,N}

  tmp_data = allocate_tmp_data( NSQECC(), size(G), size(F), precision )
  
  tmp_data[1][Base.OneTo.(size(G))...] .= G
  tmp_data[2][Base.OneTo.(size(F))...] .= F
  integralArraySQ!( tmp_data[3], tmp_data[2], (1,1), size(F) ); 
    
  _NSQECC!( tmp_data..., size(G), size(F) ); 

  destroy_fftw_plans( NSQECC(), tmp_data ); 

  return tmp_data[1], tmp_data[2]
end









"""
  Out-of-place implementation of NSQECC cross-correlation.
  ------------------------------------------------------------------------
  
  This function allows to apply NSQECC cross-correlation as a pattern matching
  operation of two inputs for general applications, not only PIV.

  In constrast with the PIV-specific implementation, the size of F is not
  size(G) .+ 2 .* search_marg, which is always an even number. Therefore, 
  we can't assume anything about the shape of the input data.
  
  This influences the construction of the r2c/c2r plans and the circshifting
  of the cross-correlation in the frequency domain. 
"""
function _NSQECC( G::Array{T,N}, F::Array{T,N}; precision=32, silent=true ) where {T,N}

    silent || println( "running non-piv NSQECC cross-correlation" ); 

    tmp_data = allocate_tmp_data_nopiv( NSQECC(), size(G), size(F), precision )
    
    tmp_data[1][Base.OneTo.(size(G))...] .= G
    tmp_data[2][Base.OneTo.(size(F))...] .= F
    integralArraySQ!( tmp_data[3], tmp_data[2], (1,1), size(F) ); 
    
    _NSQECC_nopiv!( tmp_data..., size(G), size(F) ); 

    destroy_fftw_plans( NSQECC(), tmp_data )

    return tmp_data[1], tmp_data[2]
end

function allocate_tmp_data_nopiv( ::NSQECC, isize::Dims{N}, ssize::Dims{N}, precision=32 ) where {N}
  T          = ( precision == 32 ) ? Float32 : Float64; 
  csize      = isize .+ ssize .- 1; 
  r2cpad     = 1 + iseven( csize[1] )         
  pad_csize  = csize .+ ( r2cpad, zeros(Int,N-1)... );  
  pad_inter  = zeros( T, pad_csize ); 
  pad_search = zeros( T, pad_csize ); 
  int_search = zeros( T, ssize .+ 1 ); 
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );  
  c2r_plan   = inplace_c2r_plan( pad_search, csize );
  return ( pad_inter, pad_search, int_search, r2c_plan, c2r_plan ) 
end

function _NSQECC_nopiv!( pad_G::Array{T,2}, pad_F::Array{T,2}, int2_F::Array{T,2}, r2c, c2r, size_G, size_F ) where {T<:AbstractFloat}
  sumG2 = 0.0
  @inbounds for x in 1:size_G[2], y in 1:size_G[1]
    sumG2 += pad_G[y,x]^2
  end
  sqrtsumG2 = sqrt( sumG2 ) 
  _FFTCC_nopiv!( pad_G, pad_F, r2c, c2r, size_G, size_F ); 
  pad_G     .= 0.0;
  TLs, BRs   = size_F .- size_G .+ 2, size_F .+ 1; 
  TLx, BRx   = TLs[2], BRs[2];
  for x in size_G[2]:size_F[2];    TLx -= 1;  BRx -= 1; 
    TLy, BRy = TLs[1], BRs[1];
    for y in size_G[1]:size_F[1];  TLy -= 1;  BRy -= 1; 
      sumF2 = abs( integralArea( int2_F, (TLy,TLx), (BRy,BRx) ) )
      sumGF = pad_F[ y, x ]
      L2    = ( sumF2 + sumG2 - 2*sumGF ) / ( sqrtsumG2 * sqrt(sumF2) )
      pad_G[ y, x ] = 1 / ( 1 + L2 )
  end end
end

function _NSQECC_nopiv!( pad_G::Array{T,3}, pad_F::Array{T,3}, int2_F::Array{T,3}, r2c, c2r, size_G, size_F ) where {T<:AbstractFloat}
  sumG2 = 0.0
  @inbounds for z in 1:size_G[3], x in 1:size_G[2], y in 1:size_G[1]
    sumG2 += pad_G[y,x,z]^2
  end
  sqrtsumG2 = sqrt( sumG2 ) 
  _FFTCC_nopiv!( pad_G, pad_F, r2c, c2r, size_G, size_F ); 
  pad_G     .= 0.0;
  ovp0, ovp1 = size_G, size(pad_F) .- size_G .+ 1; 
  TLFs, BRBs = size_F .- size_G, size_F .+ 1; 
  TLFz, BRBz = TLFs[3], BRBs[3]
  for z in ovp0[3]:ovp1[3];      TLFz -= 1;  BRBz -= 1; 
    TLFx, BRBx = TLFs[2], BRBs[2];
    for x in ovp0[2]:ovp1[2];    TLFx -= 1;  BRBx -= 1; 
      TLFy, BRBy = TLFs[1], BRBs[1];
      for y in ovp0[1]:ovp1[1]-r2cpad;  TLFy -= 1;  BRBy -= 1; 
        sumF2 = abs( integralArea( int2_F, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) ) )
        sumGF = pad_F[ y, x, z ]
        L2    = ( sumF2 + sumG2 - 2*sumGF ) / ( sqrtsumG2 * sqrt(sumF2) )
        pad_G[ y, x, z ] = 1 / ( 1 + L2 )
  end end end
end