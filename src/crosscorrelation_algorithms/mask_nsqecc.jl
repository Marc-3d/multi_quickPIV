
"""
  This function is the workhorse of PIV. It computes in-place cross-correlation of
a pair of interrogation/search regions and computes a displacement vector.  

  This particular implementation corresponds to the masked NSQECC, where
  the user can provide a mask for the interrogation region, and only the
  masked pixels will be considered in the cross-correlation. 
"""
function displacement_from_crosscorrelation( ::mask_NSQECC, G, F, mask, TLF, scale, 
                                                            pivparams, tmp_data )  

  # COPYING INPUT DATA INTO THEIR RESPECTIVE PADDED ARRAYS AND COMPUTING INTEGRAL ARRAYS
  prepare_inputs!( mask_NSQECC(), G, F, mask, TLF, scale, pivparams, tmp_data )

  # COMPUTING maske_NSQECC MATRIX INPLACE (ON PAD_G)
  _mask_NSQECC!( tmp_data..., scale, pivparams, pivparams.ovp_th )  

  # COMPUTE DISPLACEMENT
  tmp_data[1][ Base.UnitRange.(   size(tmp_data[1]) .- (3,0,0), size(tmp_data[1]) )... ] .= 0.0
  displacement = gaussian_displacement( tmp_data[1], scale, pivparams )
  
  return displacement
end

"""
  In order to compute masked_NSQECC we need: 
    [1] Array{Float32/64,N}: padded masked_interr_region
    [2] Array{Float32/64,N}: padded search_region        
    [3] Array{Float32/64,N}: padded   mask_region   
    [4] Array{Float32/64,N}: padded search_region .^2    
    [5] Array{Float32/64,N}: integral array of masked_interr_region .^2
    [6] Array{Float32/64,N}: integral array of mask     
    [7] FFT_plan_forward   : forward r2c FFT plan ( inplace )
    [8] FFT_plan_inverse   : inverse c2r FFT plan ( inplace )
"""
function allocate_tmp_data( ::mask_NSQECC, scale, pivparams::PIVParameters, precision=32 )
  return allocate_tmp_data( mask_NSQECC(), _isize(pivparams,scale), _ssize(pivparams,scale), precision )
end

function allocate_tmp_data( ::mask_NSQECC, inter_size::Dims{N}, search_size::Dims{N}, precision=32 ) where {N}

  corr_size     = inter_size .+ search_size .- 1 .+ 1;
  pad_corr_size = corr_size .+ ( 2, zeros(Int,N-1)... ); 
  T             = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter     = zeros( T, pad_corr_size ); 
  pad_search    = zeros( T, pad_corr_size ); 
  pad_search2   = zeros( T, pad_corr_size );
  pad_mask      = zeros( T, pad_corr_size );
  int_inter2    = zeros( T, inter_size .+ 1 );
  int_mask      = zeros( T, inter_size .+ 1 );
  r2c_plan      = inplace_r2c_plan( pad_inter , corr_size );  
  c2r_plan      = inplace_c2r_plan( pad_search, corr_size );
  return ( pad_inter, pad_search, pad_mask, pad_search2, int_inter2, int_mask, r2c_plan, c2r_plan )  
end

"""
  Each interrogation area is defined by its top-left-corner (TLF) and the interrogation 
  size. Each search area is defined by the interrogation area (TLF + interSize) and the
  search margin around the interrogation area. With this information we can copy the 
  interrogation/search regions into the padded arrays.
  
  The function below also populates the integral arrays for G .* mask and mask, 
  from the corresponding patch ( TLF + interSize ) of G and mask. 
"""
function prepare_inputs!( ::mask_NSQECC, G, F, mask, TLF, scale, pivparams::PIVParameters, tmp_data )
  prepare_inputs!( mask_NSQECC(), G, F, mask, TLF, _isize(pivparams,scale), _smarg(pivparams,scale), tmp_data )
end

function prepare_inputs!( ::mask_NSQECC, G, F, mask, TLF, size_G, marg_F, tmp_data )

  copy_inter_masked!(   tmp_data[1], G, mask, TLF, size_G );   
  copy_search_region!(  tmp_data[2],    F   , TLF, size_G, marg_F );   
  copy_inter_region!(   tmp_data[3],  mask  , TLF, size_G );  
  copy_search_squared!( tmp_data[4],    F   , TLF, size_G, marg_F );

  integralArraySQ!( tmp_data[5], G, mask, TLF, size_G )
  integralArray!(   tmp_data[6],  mask  , TLF, size_G )
end

"""
  Destroy FFTW plans from tmp_data
"""
function destroy_fftw_plans( ::mask_NSQECC, tmp_data )
  fftw_destroy_plan( tmp_data[7] )
  fftw_destroy_plan( tmp_data[8] )
  fftw_cleanup()
end

"""
  Inplace implementation of masked_NSQECC. 
  
  Computing the sum of L2 differences between masked G and F is very similar to
  implementation in NSQECC. However, sumF2 in standard NSQECC can be computed
  with integral arrays, because the overlapping region between G and F is always
  a rectangular region. When applying a mask on G, the overlapping elements 
  between G and F have the shape of the mask at each translation, so sumF2 needs
  to be computed with a cross-correlation between the mask and F.^2. 

  Apart from this, in standard NSQECC we made the simplification that sumG2 is
  constant for all displacement. In this implementation we don't make that 
  simplification... but we could if we wanted too.

  Here is the formula for decomposing NSQECC: 

    sum( (mask_G - F)^2 ) = sum( mask_G^2 ) + sum( F^2 ) - 2 * sum( mask_G * F )

  > sum( F^2 ) and sum( mask_G * F ) are computed with cross-correlations. 
  > sum( mask_G^2 ) is computed with integral arrays. 
"""
function _mask_NSQECC!( pad_G, pad_F, pad_mask, pad_F2, int_G2, int_mask, 
                        r2c, c2r, scale, pivparams::PIVParameters, ovp_th )
                        
  _mask_NSQECC!( pad_G, pad_F, pad_mask, pad_F2, int_G2, int_mask, r2c, c2r,
                 _isize(pivparams,scale), _ssize(pivparams,scale), ovp_th )
end

function _mask_NSQECC!( pad_G::T , pad_F::T , pad_mask::T, 
                        pad_F2::T, int_G2::T, int_mask::T, 
                        r2c, c2r, size_G::Dims{2}, size_F::Dims{2},
                        ovp_th=0.5 ) where {T<:AbstractArray{<:AbstractFloat,2}}

    # COMPUTING THE TOTAL NUMBER OF 1'S IN THE MASK
    maxN = 0
    for x in 1:size_G[2], y in 1:size_G[1]
      @inbounds maxN += pad_mask[y,x]
    end

    # CROSSCORRELATE (mask .* G) ⋆ F, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( mask_G * F )
    _FFTCC!( pad_G   , pad_F , r2c, c2r, size_G, size_F );

    # CROSCORRELATE mask ⋆ F^2, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( F^2 )
    _FFTCC!( pad_mask, pad_F2, r2c, c2r, size_G, size_F );

    pad_G .= 0.0; 
    # USING INTEGRAL ARRAYS TO COMPUTE N & SUMG2 & NSQECC FORE EACH TRANSLATION
    TLx, BRx = 1, 0;
    for x in 1:size(pad_F,2)-1      
      TLx += Int(x > size_F[2]); 
      BRx += Int(x < size_G[2]); 

      TLy, BRy = 1, 0;
      for y in 1:size(pad_F,1)-3
        TLy += Int(y > size_F[1]); 
        BRy += Int(y < size_G[1]); 

        N = integralArea( int_mask, (TLy,TLx), (BRy,BRx) );
        ( N/maxN < ovp_th ) && ( continue; ) # Too little overlap
        
        sumG2 = abs( integralArea( int_G2, (TLy,TLx), (BRy,BRx) ) );
        sumF2 = abs( pad_F2[y,x] ); 
        sumGF = pad_F[y,x]
        L2    = ( sumG2 + sumF2 - 2*sumGF )/( sqrt( sumG2 )*sqrt( sumF2 ) )
        pad_G[y,x] = 1/( 1 + L2 ); 
        
    end end
end

# 3D version ( 3 for loops instead of 2 ). 
function _mask_NSQECC!( pad_G::T , pad_F::T , pad_mask::T, 
                        pad_F2::T, int_G2::T, int_mask::T, 
                        r2c, c2r, size_G::Dims{3}, size_F::Dims{3},
                        ovp_th=0.5 ) where {T<:AbstractArray{<:AbstractFloat,3}}

    maxN = 0
    for z in 1:size_G[3], x in 1:size_G[2], y in 1:size_G[1]
      @inbounds maxN += pad_mask[y,x,z]
    end

    _FFTCC!( pad_G   , pad_F , r2c, c2r, size_G, size_F );
    _FFTCC!( pad_mask, pad_F2, r2c, c2r, size_G, size_F );
    
    pad_G .= 0.0; 
    
    TLFz, BRBz = 1, 0; 
    for z in 1:size(pad_F,3)-1;        TLFz += Int(z > size_F[3]); BRBz += Int(z < size_G[3]); 
      TLFx, BRBx = 1, 0; 
      for x in 1:size(pad_F,2)-1;      TLFx += Int(x > size_F[2]); BRBx += Int(x < size_G[2]); 
        TLFy, BRBy = 1, 0;
        for y in 1:size(pad_F,1)-3;    TLFy += Int(y > size_F[1]); BRBy += Int(y < size_G[1]); 

          N = integralArea( int_mask, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) )
          ( N/maxN < ovp_th ) && ( continue; ) 

          sumG2 = abs( integralArea( int_G2, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) ) )
          sumF2 = abs( pad_F2[y,x,z] )
          sumGF = pad_F[y,x,z]; 
          L2    =  ( sumG2 + sumF2 - 2*sumGF )/( sqrt( sumG2 )*sqrt( sumF2 ) );
          pad_G[y,x,z] = 1/( 1 + L2 ); 

    end end end
end

#=
  OUT-OF-PLACE NSQECC FOR DEBUGGING. 
  
  IT ASSUMES THAT size(G) .+ size(F) .- 1 IS ODD. IT WILL FAIL IF THIS IS NOT THE CASE.
=#
function _mask_NSQECC_piv( G::Array{T,N}, F::Array{T,N}, mask::Array{<:Real,N}, ovp_th=0.5) where {T,N}

    tmp_data = allocate_tmp_data( mask_NSQECC(), size(G), size(F), sizeof(T)*8 )

    copy_inter_masked!( tmp_data[1], G, mask, (1,1), size(G) ); 
    copy_inter_region!( tmp_data[2],    F   , (1,1), size(F) );
    copy_inter_region!( tmp_data[3],   mask , (1,1), size(G) ); 
    copy_inter_squared!( tmp_data[4],   F   , (1,1), size(F) ); 

    integralArraySQ!( tmp_data[5], G, mask ); 
    integralArray!( tmp_data[6], mask ); 

    _mask_NSQECC!( tmp_data..., size(G), size(F), ovp_th ); 

    destroy_fftw_plans( mask_NSQECC(), tmp_data ); 

    return tmp_data[1], tmp_data[2]
end









"""
  Out-of-place implementation of mask_NSQECC cross-correlation.
  ------------------------------------------------------------------------
  
  This function allows to apply mask_NSQECC cross-correlation as a pattern matching
  operation of two inputs for general applications, not only PIV.

  In constrast to NSQECC, a mask is applied to G, and only the masked pixels are 
  considered for computing L2 similarities. This has two main consequences:

    > sum_F^2 has to be computed with cross-correlation and not integral array, 
      since the overlap between mask_G and F is no longer a square region. 

    > sum_maskG is not constant. 

    > we add a threshold about the proportion of masked pixels that overlap F 
      at each translation. This allows to filter translation with low overlaps.
"""
function _mask_NSQECC( G::Array{T,N}, F::Array{T,N}, mask::Array{<:Real,N}, ovp_th=0.5) where {T,N}

  tmp_data = allocate_tmp_data_nopiv( mask_NSQECC(), size(G), size(F), sizeof(T)*8 )

  copy_inter_masked!( tmp_data[1], G, mask, (1,1), size(G) ); 
  copy_inter_region!( tmp_data[2],    F   , (1,1), size(F) );
  copy_inter_region!( tmp_data[3],   mask , (1,1), size(G) ); 
  copy_inter_squared!( tmp_data[4],   F   , (1,1), size(F) ); 

  integralArraySQ!( tmp_data[5], G, mask ); 
  integralArray!( tmp_data[6], mask ); 

  _mask_NSQECC_nopiv!( tmp_data..., size(G), size(F), ovp_th ); 

  destroy_fftw_plans( mask_NSQECC(), tmp_data ); 

  return tmp_data[1], tmp_data[2]
end

function allocate_tmp_data_nopiv( ::mask_NSQECC, inter_size::Dims{N}, search_size::Dims{N}, precision=32 ) where {N}

  corr_size     = inter_size .+ search_size .- 1;
  pad_corr_size = corr_size .+ ( 2, zeros(Int,N-1)... ); 
  T             = ( precision == 32 ) ? Float32 : Float64; 
  pad_inter     = zeros( T, pad_corr_size ); 
  pad_search    = zeros( T, pad_corr_size ); 
  pad_search2   = zeros( T, pad_corr_size );
  pad_mask      = zeros( T, pad_corr_size );
  int_inter2    = zeros( T, inter_size .+ 1 );
  int_mask      = zeros( T, inter_size .+ 1 );
  r2c_plan      = inplace_r2c_plan( pad_inter , corr_size );  
  c2r_plan      = inplace_c2r_plan( pad_search, corr_size );
  return ( pad_inter, pad_search, pad_mask, pad_search2, int_inter2, int_mask, r2c_plan, c2r_plan )  
end

function _mask_NSQECC_nopiv!( pad_G::T , pad_F::T , pad_mask::T, 
                              pad_F2::T, int_G2::T, int_mask::T, 
                              r2c, c2r, size_G::Dims{2}, size_F::Dims{2},
                              ovp_th=0.5 ) where {T<:AbstractArray{<:AbstractFloat,2}}

    # COMPUTING THE TOTAL NUMBER OF 1'S IN THE MASK
    maxN = 0
    for x in 1:size_G[2], y in 1:size_G[1]
      @inbounds maxN += pad_mask[y,x]
    end

    # CROSSCORRELATE (mask .* G) ⋆ F, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( mask_G * F )
    _FFTCC_nopiv!( pad_G, pad_F, r2c, c2r, size_G, size_F );

    # CROSCORRELATE mask ⋆ F^2, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( F^2 )
    _FFTCC_nopiv!( pad_mask, pad_F2, r2c, c2r, size_G, size_F );

    pad_G .= 0.0; 
    # USING INTEGRAL ARRAYS TO COMPUTE N & SUMG2 & NSQECC FORE EACH TRANSLATION
    TLx, BRx = 1, 0;
    for x in 1:size(pad_F,2)-1      
      TLx += Int(x > size_F[2]); 
      BRx += Int(x < size_G[2]); 

      TLy, BRy = 1, 0;
      for y in 1:size(pad_F,1)-3
        TLy += Int(y > size_F[1]); 
        BRy += Int(y < size_G[1]); 

        N = integralArea( int_mask, (TLy,TLx), (BRy,BRx) );
        ( N/maxN < ovp_th ) && ( continue; ) # Too little overlap
        
        sumG2 = abs( integralArea( int_G2, (TLy,TLx), (BRy,BRx) ) );
        sumF2 = abs( pad_F2[y,x] ); 
        sumGF = pad_F[y,x]
        L2    = ( sumG2 + sumF2 - 2*sumGF )/( sqrt( sumG2 )*sqrt( sumF2 ) )
        pad_G[y,x] = 1/( 1 + L2 ); 
        
    end end
end

# 3D version ( 3 for loops instead of 2 ). 
function _mask_NSQECC_nopiv!( pad_G::T , pad_F::T , pad_mask::T, 
                              pad_F2::T, int_G2::T, int_mask::T, 
                              r2c, c2r, size_G::Dims{3}, size_F::Dims{3},
                              ovp_th=0.5 ) where {T<:AbstractArray{<:AbstractFloat,3}}

    maxN = 0
    for z in 1:size_G[3], x in 1:size_G[2], y in 1:size_G[1]
      @inbounds maxN += pad_mask[y,x,z]
    end

    _FFTCC_nopiv!( pad_G, pad_F, r2c, c2r, size_G, size_F );
    _FFTCC_nopiv!( pad_mask, pad_F2, r2c, c2r, size_G, size_F );
    pad_G .= 0.0; 
    
    TLFz, BRBz = 1, 0; 
    for z in 1:size_G[3];        TLFz += Int(z > size_F[3]); BRBz += Int(z < size_G[3]); 
      TLFx, BRBx = 1, 0; 
      for x in 1:size_G[2];      TLFx += Int(x > size_F[2]); BRBx += Int(x < size_G[2]); 
        TLFy, BRBy = 1, 0;
        for y in 1:size_G[1];    TLFy += Int(y > size_F[1]); BRBy += Int(y < size_G[1]); 

          N = integralArea( int_mask, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) )
          ( N/maxN < ovp_th ) && ( continue; ) 

          sumG2 = abs( integralArea( int_G2, (TLFy,TLFx,TLFz), (BRBy,BRBx,BRBz) ) )
          sumF2 = abs( pad_F2[y,x,z] )
          sumGF = pad_F[y,x,z]; 
          L2    =  ( sumG2 + sumF2 - 2*sumGF )/( sqrt( sumG2 )*sqrt( sumF2 ) );
          pad_G[y,x,z] = 1/( 1 + L2 ); 

    end end end
end