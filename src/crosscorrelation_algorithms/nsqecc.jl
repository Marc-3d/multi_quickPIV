
"""
  This function is the workhorse of PIV. It computes in-place cross-correlation of
  a pair of interrogation/search regions and computes a displacement vector. 

  This function corresponds to NSQECC, which introduces L2 as the similarity measure
  at each translation of the cross-correlation matrix. L2 can be computed with 1 
  cross-correlation, and integral areas to compute sum_F^2 at each translation.
"""
function displacement_from_crosscorrelation( ::NSQECC, G, F, TLF, scale, pivparams::PIVParameters, tmp_data )  

  # COPYING INPUT DATA INTO THEIR RESPECTIVE PADDED ARRAYS.
  prepare_inputs!( NSQECC(), G, F, TLF, scale, pivparams, tmp_data )

  # COMPUTING NSQECC MATRIX INPLACE (ON PAD_G)
  _NSQECC!( tmp_data..., scale, pivparams )  

  # COMPUTE DISPLACEMENT
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
  return allocate_tmp_data( NSQECC(), _isize(pivparams, scale), _ssize(pivparams, scale), precision )
end

function allocate_tmp_data( ::NSQECC, inter_size::Dims{N}, search_size::Dims{N}, precision=32 ) where {N}

  csize      = inter_size .+ search_size .- 1 .+ 1; 
  pad_csize  = csize .+ ( 2, zeros(Int,N-1)... );    
  T          = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter  = zeros( T,   pad_csize  ); 
  pad_search = zeros( T,   pad_csize  ); 
  int_search = zeros( T, search_size .+ 1 ); 
  r2c_plan   = inplace_r2c_plan( pad_inter , csize );  
  c2r_plan   = inplace_c2r_plan( pad_search, csize );

  return ( pad_inter, pad_search, int_search, r2c_plan, c2r_plan )  
end

"""
  Each interrogation area is defined by its top-left-corner (TLF) and the interrogation 
  size. Each search area is defined by the interrogation area (TLF + interSize) and the
  search margin around the interrogation area. With this information we can copy the 
  interrogation/search regions into the padded arrays.

  In addition, NSQECC requires computing the integral array from the search region.^2.
"""
function prepare_inputs!( ::NSQECC, G, F, TLF, scale, pivparams::PIVParameters, tmp_data )
  prepare_inputs!( NSQECC(), G, F, TLF, _isize(pivparams,scale), _smarg(pivparams,scale), tmp_data ) 
end

function prepare_inputs!( ::NSQECC, G, F, TLF, size_G::Dims{N}, marg_F::Dims{N}, tmp_data ) where {N}

  copy_inter_region!( tmp_data[1], G, TLF, size_G );          
  copy_search_region!( tmp_data[2], F, TLF, size_G, marg_F );
  integralArraySQ!( tmp_data[3], tmp_data[2], ones(Int64,N), size_G .+ 2 .* marg_F ); 
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
  have already been computed in (1).

  NOTE: We only consider translations with full overlap between G and F, because it is
  extrapolate regions partial overlap when using the L2 similarity. 
"""
function _NSQECC!( pad_G, pad_F, int2_F, r2c, c2r, scale, pivparams::PIVParameters )
  return _NSQECC!( pad_G, pad_F, int2_F, r2c, c2r, _isize(pivparams,scale), _ssize(pivparams,scale) );
end

function _NSQECC!( pad_G::Array{T,2}, pad_F::Array{T,2}, int2_F::Array{T,2}, r2c, c2r, size_G::Dims{2}, size_F::Dims{2} ) where {T<:AbstractFloat}

  # COMPUTING INTERROGATION REGION CONSTANTS, BEFORE ANY IN-PLACE FFT. 
  sumG2 = 0.0
  for x in 1:size_G[2], y in 1:size_G[1]
    @inbounds sumG2 += pad_G[y,x]^2
  end
  sqrtsumG2 = sqrt( sumG2 ) 

  # COMPUTING REAL-DATA IN-PLACE CROSS-CORRELATION MATRIX, G ⋆ F, ON PAD_F.
  _FFTCC!( pad_G, pad_F, r2c, c2r, size_G, size_F ); 

  # COMPUTING NSQECC IN-PLACE (only for full-overlapping translations)
  pad_G .= 0.0;

  TLx = size_F[2] - size_G[2] + 1 + 1;
  BRx = size_F[2] + 1;
  for x in 1+size_G[2]-1:size(pad_F,2)-size_G[2]+1-1     
    TLx -= 1;
    BRx -= 1; 

    TLy = size_F[1] - size_G[1] + 1 + 1;
    BRy = size_F[1] + 1
    for y in 1+size_G[1]-1:size(pad_F,1)-size_G[1]+1-3
      TLy -= 1; 
      BRy -= 1; 

      # Float precision might produce small negative values. Using abs() to avoid sqrt(neg) errors. 
      sumF2 = abs( integralArea( int2_F, (TLy,TLx), (BRy,BRx) ) )
      sumGF = pad_F[ y, x ]
      num   = sumF2 + sumG2 - 2*sumGF
      den   = sqrtsumG2 * sqrt(sumF2)
      pad_G[ y, x ] = 1 / ( 1 + num/den )
  end end
end

function _NSQECC!( pad_G::Array{T,3}, pad_F::Array{T,3}, int2_F::Array{T,3}, r2c, c2r, size_G::Dims{3}, size_F::Dims{3} ) where {T<:AbstractFloat}

  sumG2 = 0.0
  @inbounds for z in 1:size_G[3], x in 1:size_G[2], y in 1:size_G[1]
    sumG2 += pad_G[y,x,z]^2
  end
  sqrtsumG2 = sqrt( sumG2 ) 

  _FFTCC!( pad_G, pad_F, r2c, c2r, size_G, size_F ); 

  pad_G     .= 0.0;
  ovp0, ovp1 = size_G, size(pad_F) .- size_G .+ 1; 
  TLFs, BRBs = size_F .- size_G .+ 2, size_F .+ 1; 

  TLFz, BRBz = TLFs[3], BRBs[3]
  for z in ovp0[3]:ovp1[3]-1;      TLFz -= 1;  BRBz -= 1; 
    TLFx, BRBx = TLFs[2], BRBs[2];
    for x in ovp0[2]:ovp1[2]-1;    TLFx -= 1;  BRBx -= 1; 
      TLFy, BRBy = TLFs[1], BRBs[1];
      for y in ovp0[1]:ovp1[1]-3;  TLFy -= 1;  BRBy -= 1; 
        sumF2 = abs( integralArea( int2_F, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) ) )
        sumGF = pad_F[ y, x, z ]
        L2    = ( sumF2 + sumG2 - 2*sumGF ) / ( sqrtsumG2 * sqrt(sumF2) )
        pad_G[ y, x, z ] = 1 / ( 1 + L2 )
  end end end
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

function _NSQECC_nopiv!( pad_G::Array{T,3}, pad_F::Array{T,3}, int2_F::Array{T,2}, r2c, c2r, size_G, size_F ) where {T<:AbstractFloat}
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