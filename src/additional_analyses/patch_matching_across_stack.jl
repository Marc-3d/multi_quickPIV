"""
    Applied for aligning cryoEM stacks. 
    
    Input: 
        
    3D tilt stack, where each slice is an Cryo image, and each slice
    corresponds to a different tilt angle of the sample. In Julia words: 
        size( input, 1 ) = height; 
        size( input, 2 ) = width; 
        size( input, 3 ) = number_of_tilts = T 

    Goal: 


    The initial slice at "tilt_0" is divided into a grid of patches, like in
    PIV, according to the paramters "interSize" and "overlap/step". 
    
    Given these patches, the goal is to track each one of the patche from slice
    "tilt_0" across the desired slices "tilt_0:tilt_1". This creates a track or 
    trajectory for each initial patch  across the desired set of tilts. 

    Output: 
    
    A table containing N*T rows and 3 columns, where N is the number of patches
    at the first slice, and T is the number of tilts (the third dimension of the
    stack). The 2 first columns corresponds to the X, Y coordinates of the patches
    and the 3rd column records the slice index. 
    
    For instance, the first T rows corresponds the positions of the first patch 
    accross the Tilts: (X1_0,Y1_0,tilt_0), ...  (X1_T,Y1_T,tilt_1). The second T rows 
    are the positions of the second patch across the T tilts: (X2_0,Y2_0,tilt_0), ...
    (X2_T,Y2_T,tilt_1). 
"""

function patch_matching_across_tilts( stack::Array{DT,3}, 
                                      pivparams::PIVParameters; 
                                      precision=32, 
                                      tilt_0=1,
                                      tilt_1=size(stack,3), 
                                      skip=(0,0)
                                    ) where {DT<:Real}

    pivparams.ndims = 2;
    inp_size = size( stack )[1:2]; 

    # DIMENIONS OF THE GRID OF PATCHES ON THE FIRST SLICE
    vf_size       = get_vectorfield_size( inp_size, pivparams, 1 )
    patch_size    = _isize( pivparams ); 
    patch_spacing = _step( pivparams );
    patch_center  = div.( patch_size .+ 1, 1 );

    # TEMPORAL ARRAYS FOR IN-PLACE CROSSCORRELATIONS
    tmp_data = allocate_tmp_data( pivparams.corr_alg, 1, pivparams, precision )

    # OUTPUT TABLE, CONTAINING THE TRACK FOR EACH INITIAL PATCH
    N, T     = prod( vf_size ), length( tilt_0:tilt_1-1 ); 
    out_type = ( precision == 32 ) ? Float32 : Float64; 
    output   = zeros( out_type, N*T, 3 ); 
    row_idx  = 1; #

    # For each patch
    for vf_idx in 1:prod( vf_size )

        # Cartesian coordinates of the patch
        vf_coords = get_vf_coords( vf_idx, inp_size, pivparams, 1 )

        # It is usually desired to skip patches that are close to the edges. We achieve that by
        # skipping patches that are too close to the limits of the vector field.
        if any( (vf_coords .- 1) .< skip[1] ) || any( (vf_size .- vf_coords) .< skip[2] )
            row_idx += T; 
            continue
        end
        
        # COMPUTING COORDINATES FOR THE CURRENT PAIR OF INTERROGATION/SEARCH REGION
        coord_data = get_interr_and_search_coordinates( vf_idx, vf_size, inp_size, (0,0), 1, pivparams );
        
        offset = Int[ 0, 0 ]
        patch_position = patch_center .+ patch_spacing .* ( vf_coords .- 1 ); 

        # For each tilt 
        for tilt in tilt_0:tilt_1-1

            # COPYING INTERROGATION/SEARCH REGIONS INTO PADDED ARRAYS FOR FFT
            input1 = view( stack, :, :,  tilt  ); 
            input2 = view( stack, :, :, tilt+1 ); 
            prepare_inputs!( pivparams.corr_alg, input1, input2, coord_data, tmp_data );

            # DEALING WITH POSSIBLE OUT-OF-BOUNDS EXTENT OF THE INTERROGATION REGION
            IR_TLF    , IR_BRB     = coord_data[1], coord_data[2]; 
            IR_TLF_off, IR_BRB_off = coord_data[7], coord_data[8];

            tmp_data[1] .= 0
            pad_range = UnitRange.( (1,1).+IR_TLF_off, _isize(pivparams).-IR_BRB_off )
            tmp_data[1][ pad_range... ] .= input1[ UnitRange.( IR_TLF, IR_BRB )... ]

            # COMPUTING CROSS-CORRELATION MATRIX
            crosscorrelation!( pivparams.corr_alg, 1, pivparams, tmp_data ); 

            # COMPUTING DISPLACEMENT FROM MAXIMUM PEAK OF CC + GAUSSIAN REFINEMENT
            displacement = gaussian_displacement( tmp_data[1], 1, pivparams, coord_data, tmp_data )

            # EXTENDING THE TRACK
            output[ row_idx, 1:2 ] .= patch_position
            output[ row_idx,  3  ] = tilt

            # updating variables 
            patch_position = patch_position .- displacement; 
            offset = round.( Int, displacement ); 
            row_idx += 1
        end
    end
    
    destroy_fftw_plans( pivparams.corr_alg, tmp_data )

    return output

end