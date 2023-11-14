"""
  This particular implementation corresponds to the masked NSQECC, where
  the user can provide a mask for the interrogation region, and only the
  masked pixels will be considered in the cross-correlation. 

  The mask can be a boolean (0,1) or weights 0<w<1. This means that
  the mask can highlight or undermine certain pixels/structures in the 
  interrogation input. 
"""

function crosscorrelation!( ::mask_NSQECC, 
                             scale, 
                             pivparams::PIVParameters, 
                             tmp_data
                          )

  _mask_NSQECC!( tmp_data..., scale, pivparams )  
  return nothing
end


"""
  In order to compute masked_NSQECC we need: 
    [1] Array{Float32/64,N}: padded masked_interr_region
    [2] Array{Float32/64,N}: padded search_region        
    [3] Array{Float32/64,N}: padded   mask_region   
    [4] Array{Float32/64,N}: padded search_region .^2      
    [5] FFT_plan_forward   : forward r2c FFT plan ( inplace )
    [6] FFT_plan_inverse   : inverse c2r FFT plan ( inplace )
    [7] csize              : we keep track of the cross-correlation pad, after padding
"""

function allocate_tmp_data( ::mask_NSQECC, scale, pivparams::PIVParameters, precision=32 )

  return allocate_tmp_data( mask_NSQECC(), _isize(pivparams,scale), _ssize(pivparams,scale), 
                            precision=precision, 
                            unpadded=pivparams.unpadded, 
                            good_pad=pivparams.good_pad, 
                            odd_pad=pivparams.odd_pad
                          )
end

function allocate_tmp_data( ::mask_NSQECC, 
                            isize::Dims{N}, 
                            ssize::Dims{N}; 
                            precision=32,
                            unpadded=true,
                            good_pad=false,
                            odd_pad=true
                          ) where {N}

  csize1, r2c_pad, corr_pad = compute_csize_and_paddings( isize, ssize, 
                                                          unpadded=unpadded, 
                                                          good_pad=good_pad, 
                                                          odd_pad=odd_pad )

  pad_csize   = csize1 .+ r2c_pad;
  pad_ctype   = ( precision == 32 ) ? Float32 : Float64
  pad_inter   = zeros( pad_ctype, pad_csize ); 
  pad_search  = zeros( pad_ctype, pad_csize ); 
  pad_search2 = zeros( pad_ctype, pad_csize );
  pad_mask    = zeros( pad_ctype, pad_csize );
  r2c_plan    = inplace_r2c_plan( pad_inter , csize1 );  
  c2r_plan    = inplace_c2r_plan( pad_search, csize1 );

  return ( pad_inter, pad_search, pad_mask, pad_search2, r2c_plan, c2r_plan, csize1 )  
end


"""
  Copying the interrogation and search regions into the padded arrays for the FFTs.
  In addition, we populate an integral array with the values^2 of the search region, 
  which will allow us to compute the L2 errors efficiently for each translation.
"""

function prepare_inputs!( ::mask_NSQECC, F, G, mask, 
                                         coord_data,
                                         tmp_data )

  copy_inter_masked!(   tmp_data[1], F, mask, coord_data );   
  copy_search_region!(  tmp_data[2],    G   , coord_data );   
  copy_inter_region!(   tmp_data[3],  mask  , coord_data );  
  copy_search_squared!( tmp_data[4],    G   , coord_data );

  return nothing
end


"""
  Destroy FFTW plans from tmp_data
"""

function destroy_fftw_plans( ::mask_NSQECC, tmp_data )
  fftw_destroy_plan( tmp_data[5] )
  fftw_destroy_plan( tmp_data[6] )
  fftw_cleanup()
end


"""
  Inplace implementation of masked_NSQECC. 
  
  Computing the sum of L2 differences between masked F and G is very similar to
  the implementation in NSQECC. However, sumG2 in standard NSQECC can be computed
  with integral arrays, because the overlapping region between G and F is always
  a rectangular region. When applying a mask on F, the overlapping elements 
  between F and G have the shape of the mask at each translation, so sumG2 needs
  to be computed with a cross-correlation between the mask and G.^2. 

  Apart from this, in standard NSQECC we made the simplification that sumF2 is
  constant for all displacement, since we only consider fully-overlapping 
  translations. In masked_NSQECC we also make this simplification...

  Here is the formula for decomposing NSQECC: 

    sum( (mask_F - G)^2 ) = sum( mask_F^2 ) + sum( G^2 ) - 2 * sum( mask_F * G )

  > sum( G^2 ) and sum( mask_G * F ) are computed with cross-correlations. 
  > sum( mask_F^2 ) is a constant, and not computed with integral arrays. 

  sum( wi(F - G)^2 ) = sum( w_i .* mask_F^2 ) + sum( w_i .* G^2 ) - 2 * sum( w_i * mask_F * G )

"""

function _mask_NSQECC!( pad_maskF, pad_G, 
                        pad_mask, pad_G2, 
                        r2c, c2r, csize, 
                        scale, pivparams::PIVParameters )
                        
  _mask_NSQECC!( pad_maskF, pad_G, 
                 pad_mask, pad_G2,
                 r2c, c2r, csize,
                 _isize(pivparams,scale), _ssize(pivparams,scale), 
                 pivparams.ovp_th )
end

function _mask_NSQECC!( pad_maskF::Array{T,N}, pad_G::Array{T,N}, 
                        pad_mask::Array{T,N}, pad_G2::Array{T,N}, 
                        r2c, c2r, csize, 
                        size_F::Dims{N}, size_G::Dims{N},
                        ovp_th=0.5 ) where {T<:AbstractFloat,N}

    # COMPUTING THE TOTAL NUMBER OF 1'S IN THE MASK
    maxN = 0
    sumF2 = 0.0
    D = ( N == 2 ) ? 1 : size_F[3]; 
    @inbounds for z in 1:D, x in 1:size_F[2], y in 1:size_F[1]
      maxN  += pad_mask[y,x,z]
      wi     = ( pad_mask[y,x,z] > 0 ) ? pad_mask[y,x,z] : 1; 
      sumF2 += pad_maskF[y,x,z]*pad_maskF[y,x,z]/wi
    end

    # CROSSCORRELATE (mask .* F) ⋆ G, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( mask_F * G )
    _FFTCC!( pad_maskF, pad_G, r2c, c2r, csize );

    # CROSCORRELATE mask ⋆ G^2, WITH INPLACE REAL-DATA OPTIMIZATION. => sum( G^2 )
    _FFTCC!( pad_mask, pad_G2, r2c, c2r, csize );

    fftcc2masknsqecc!( pad_maskF, pad_mask, sumF2, size_F, size_G ); 

    return nothing
end


"""
  OUT-OF-PLACE NSQECC FOR DEBUGGING. 
"""

function _mask_NSQECC( F::Array{T,N}, G::Array{T,N}, mask::Array{<:Real,N};
                       precision=32,
                       ovp_th=0.5, 
                       good_pad=false, 
                       unpadded=false, 
                       odd_pad=false
                     ) where {T,N}

    tmp_data = allocate_tmp_data( mask_NSQECC(), size(F), size(G), 
                                  precision=precision, 
                                  good_pad=good_pad, 
                                  unpadded=unpadded, 
                                  odd_pad=odd_pad )

    tmp_data[1][Base.OneTo.(size(F))...] .= F .* mask
    tmp_data[2][Base.OneTo.(size(G))...] .= G
    tmp_data[3][Base.OneTo.(size(F))...] .= mask
    tmp_data[4][Base.OneTo.(size(G))...] .= G .* G

    _mask_NSQECC!( tmp_data..., size(F), size(G), ovp_th ); 

    destroy_fftw_plans( mask_NSQECC(), tmp_data ); 

    return tmp_data[1], tmp_data[2]
end
