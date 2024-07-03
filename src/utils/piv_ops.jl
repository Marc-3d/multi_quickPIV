"""
    COMPUTING VECTOR FIELD SIZE, AKA THE NUMBER OF INTERROGATOIN AREAS THAT FIT IN EACH INPUT 
    DIMENSION.
"""

function get_vectorfield_size( input_size::Dims{N}, pivparams::PIVParameters, scale=1 ) where {N}
    get_vectorfield_size( input_size, _isize( pivparams, scale ), _step(  pivparams, scale ) )
end

function get_vectorfield_size( input_size::Dims{N}, isize::Dims{N}, step::Dims{N} ) where {N}
    return length.( StepRange.( 1, step, input_size .- isize .+ 1 ) );
end


"""
    ALLOCATES MEMORY TO HOLD THE PIV VECTOR FIELD (VF) AND THE SIGNAL-TO-NOISE (SN) MATRICES.

    2D VFs ARE STORED IN A 3D MATRIX OF SIZE (2,VF_HEIGHT,VF_WIDTH): U = VF[1,:,:], V = VF[2,:,:].
    
    3D VFs ARE STORED IN A 4D MATRIX OF SIZE (3,VF_HEIGHT,VF_WIDTH,VF_DEPTH): U = VF[1,:,:,:], 
    V = VF[2,:,:,:], W = VF[3,:,:,:].
"""
function allocate_outputs( input_size::Dims{N}, pivparams::PIVParameters, precision=32 ) where {N}

    VFsize = get_vectorfield_size( input_size, pivparams, 1 );
    VFtype = ( precision == 32 ) ? Float32 : Float64; 
    VF     = zeros( VFtype, N, VFsize... );
    SN     = ( pivparams.computeSN ) ? zeros( VFtype, VFsize... ) : nothing;
    return VF, SN
end


"""
    FINDING THE TLF (TOP-LEFT-FRONT) AND BRB (BOTTOM-RIGTH-BACK) COORDINATES OF THE "i"th
    INTERROGATION REGION. 

    SOMETIMES, I WILL REPLACE BRB ---> DRB (DOWN-RIGHT-BACK), TO AVOID REPEATING LETTERS...
"""
function get_interrogation_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N}, scale, pivparams::PIVParameters ) where {N}

    vf_coords = linear2cartesian( vf_idx, vf_size ); 
    IR_TLF    = ones(Int, N) .+ ( vf_coords .- 1 ) .* _step( pivparams, scale );
    IR_BRB    = IR_TLF .+ _isize( pivparams, scale ) .- 1; 

    return IR_TLF, IR_BRB
end


"""
    FINDING THE TLF (TOP-LEFT-FRONT) AND BRB (BOTTOM-RIGTH-BACK) COORDINATES OF THE "i"th
    INTERROGATION REGION. IN ADDITION, SEARCH COORDINATES MIGHT GO OUT-OF-BOUNDS OF THE 
    INPUT ARRAYS. THE OUT-OF-BOUND EXTENDS NEED TO BE CONSIDERED WHEN COPYING THE SEARCH
    REGION INTO THE PADDED ARRAY AND WHEN FINDING THE MAXIMUM PEAK. 

    SOMETIMES, I WILL REPLACE BRB ---> DRB (DOWN-RIGHT-BACK), TO AVOID REPEATING LETTERS...
"""
function get_search_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N}, scale, pivparams::PIVParameters ) where {N}

    vf_coords  = linear2cartesian( vf_idx, vf_size ); 
    IR_TLF     = ones(Int, N) .+ ( vf_coords .- 1 ) .*  _step( pivparams, scale );
    IR_BRB     = IR_TLF .+  _isize( pivparams, scale ) .- 1; 
    SR_TLF     = max.( 1, IR_TLF .- _smarg( pivparams, scale ) ); 
    SR_BRB     = min.( input_size, IR_BRB .+ _smarg( pivparams, scale ) );
    SR_TLF_off = abs.( min.( 0, IR_TLF .- _smarg( pivparams, scale ) .- 1 ) );
    SR_BRB_off = abs.( min.( 0, input_size .- ( IR_BRB .+ _smarg( pivparams,scale) ) ) ); 

    return SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off
end


"""
    RETRUNS A TUPLE WITH ALL COORDINATE INFORMATION AOUT THE INTERROGATION AND SEARCH REGIONS.
"""
function get_interrogation_and_search_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N}, scale, pivparams::PIVParameters ) where {N}

    IR_TLF, IR_BRB = get_interrogation_coordinates( vf_idx, vf_size, input_size, scale, pivparams )
    SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off = get_search_coordinates( vf_idx, vf_size, input_size, scale, pivparams )

    return ( IR_TLF, IR_BRB, SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off )
end


"""
    USED FOR COPYING THE INPUT DATA INTO THE PADDED ARRAYS FOR FFT. EACH COPY-FUNCTION HAS
    A VERSION THAT ACCEPTS A TUPLE OF COORDINATE DATA, TO MAKE THE CODE MORE SUCCINT. 

    coord_data = [ IA_TLF, IA_BRB, SA_TLF, SA_BRB, SA_TLF_off, SA_BRB_off ].
"""

function copy_inter_region!( pad_F, F, coord_data::NTuple{N,T} ) where {N,T}
    copy_inter_region!( pad_F, F, coord_data[1], coord_data[2] )
end

function copy_inter_region!( pad_F, F, IR_TLF, IR_BRB )
    pad_F .= 0.0; 
    size_F = IR_BRB .- IR_TLF .+ 1; 
    pad_F[ Base.OneTo.(size_F)... ] .= F[ UnitRange.(IR_TLF,IR_BRB)... ]
end

function copy_search_region!( pad_G, G, coord_data::NTuple{N,T} ) where {N,T}
    copy_search_region!( pad_G, G, coord_data[3], coord_data[4], coord_data[5] )
end

function copy_search_region!( pad_G, G, SR_TLF, SR_BRB, SR_TLF_offset )

    pad_G .= 0.0
    size_G = SR_BRB .- SR_TLF .+ 1; 
    pad_coords = UnitRange.( 1 .+ SR_TLF_offset, size_G .+ SR_TLF_offset );
    pad_G[ pad_coords... ] .= G[ UnitRange.( SR_TLF, SR_BRB )... ]; 
end


"""
    USED FOR FILTERING INTERROGATION REGIONS THAT BELONG TO THE BACKGROUND. GENERALLY,
    THOSE WITH A VERY LOW MEAN INTENSITY CORRESPOND TO BACKGROUND REGIONS.
"""
function skip_inter_region( input, IR_TLF, IR_BRB, pivparams::PIVParameters )

    if pivparams.threshold < 0
        return false
    else
        inter_view = view( input, UnitRange.( IR_TLF, IR_BRB )... ); 
        return pivparams.filtFun( inter_view ) < pivparams.threshold
    end
end


"""
    Getting vector field coordinates for each displacement vector would be quite trivial...
    it is the linear index "VFidx" of the interrogatoin/search pair in multi_quickPIV.jl:29. 
    
    However, we need to account for multipass, where displacements computed at high scales 
    (4x, 3x, etc) are used to offset the sampling of search regions in the next iterations 
    (at lower scales).

    For example, a single 2x interrogation/search pair contains 4 (2x2) 1x interrogation/search
    pairs. The corresponding 2x displacement will be used to offset the search regions of the 4
    contained 1x interrogation / search pairs. 

    In addition, overlap between interrogation areas requires interpolation of the vectors 
    computed at large scales. 
"""
#= debug code 

using PyPlot
function makeplot( N, isize, step )
    nsteps = length( StepRange( 1, step, N-isize ) )
    plot   = zeros( nsteps, N );
    for n in 1:nsteps
        tlf = 1 + ( n - 1 ) * step
        brb = tlf + isize - 1
        plot[ n, tlf:brb ] .= 1
        println( n:( n + ceil( Int, (isize-1)/step ) - 1 ) )
    end
    imshow( plot )
end

=#
function get_vf_coords( inter_index, input_size::Dims{N}, pivparams::PIVParameters, scale=1 ) where {N}

    if scale == 1
        VFsize    = get_vectorfield_size( input_size, pivparams, scale );
        VFcoords  = linear2cartesian( inter_index, VFsize ); 
        return VFcoords
    end

    # VFsize, isize and istep at the lowest scale
    VFsize_1x = get_vectorfield_size( input_size, pivparams, 1 );
    isize_1x = _isize( pivparams );
    istep_1x = (pivparams.step == nothing ) ? isize_1x .- _ovp(pivparams) : _step(pivparams); 

    # VF cartesian coordinates, isize and istep from current scale
    VFsize    = get_vectorfield_size( input_size, pivparams, scale );
    VFcoords  = linear2cartesian( inter_index, VFsize ); 
    isize     = isize_1x .* scale; 
    istep     = istep_1x .* scale; 

    # image cartesian coordinates of TLF at current scale
    TLF = ones(Int,N) .+ ( VFcoords .- 1 ) .* istep

    # VF TLF coordinates at lowest scale
    VF_TLF = div.( TLF, istep_1x ) .+ 1; 

    # VF BRB coordiantes at lowest scale
    VF_BRB = VF_TLF .+ ceil.( Int, (isize.-1) ./ istep_1x ) .- 1 
    VF_BRB = min.( VF_BRB, VFsize_1x )

    return UnitRange.( VF_TLF, VF_BRB )
end

"""
    TODO: 
"""
function update_vectorfield!( VF, counts, displacement, vf_idx, input_size::Dims{N}, pivparams::PIVParameters, scale ) where {N}

    vf_coords = get_vf_coords( vf_idx, input_size, pivparams, scale )
    #println( vf_idx, vf_coords )
    for n in 1:N
        VF[n,vf_coords...] = displacement[n]
    end
    counts[vf_coords...] += 1; 
end

function interpolate_vectorfield!( VF, counts )
    for n in 1:size(VF,1)
        VF[n,UnitRange.(1,size(VF)[2:end])...] ./= counts
    end
end


"""
    GAUSSIAN SUBPIXEL APPROXIMATION.
"""

function gaussian_displacement( corr_mat::Array{T,N}, scale, pivparams::PIVParameters, coord_data, tmp_data ) where {T,N}

    # these determine offsets of full-overlapping translations
    SR_TLF_off = coord_data[5]; 
    SR_BRB_off = coord_data[6]; 
    SM         = _smarg( pivparams, scale )

    # cross-correlation maximum peak coordinates 
    if pivparams.unpadded
        # if unpadded --> only check full ovps
        center       = SM[1:N] .+ 1; 
        peak, maxval = first_fullovp_peak( corr_mat, SM, SR_TLF_off, SR_BRB_off )
    else
        # else        --> check all translation except r2c_pad
        println("ERROR: all gaussian_displacement should be unpadded.")
        csize        = tmp_data[end]
        r2c          = size( tmp_data[1] ) .- csize; 
        center       = div.( csize, 2 )
        peak, maxval = first_fullovp_peak( corr_mat, csize, SR_TLF_off, SR_BRB_off )
    end
    # println( peak, center )
    displacement = gaussian_refinement( corr_mat, peak, maxval, SM, SR_TLF_off, SR_BRB_off ) .- center

    return displacement
end

function gaussian_displacement( corr_mat, isize::Dims{N}, ssize::Dims{N} ) where {N}

    peak, maxval = first_peak( corr_mat )
    center = div.( isize, 2 ) .+ div.( ssize, 2 )
    return gaussian_refinement( corr_mat, peak, maxval ) .- center
end

function gaussian_refinement( corr_mat::Array{T,N}, peak, maxval, SM, SR_TLF_off, SR_BRB_off ) where {T,N}

    if all( peak .> 1 .+ SR_TLF_off ) && all( peak .< 1 .+ 2 .* SM .- SR_BRB_off )
        minval = minimum( corr_mat[ UnitRange.( peak .- ones(Int,N), peak .+ ones(Int,N) )... ] )
        gaussi = gaussian( corr_mat, peak, maxval, T(minval) )
        return peak .+ gaussi
    else
        return peak 
    end
end

# 3-point Gaussian subpixel 2D from the pixel neighbourhood around the max peak 
function gaussian( corr_mat::Array{T,2}, peak, maxval::T, minval::T=0 ) where {T}

    return gaussian_2D( log(1+corr_mat[peak[1]-1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1]+1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]-1]-minval), 
                        log(1+corr_mat[peak[1],peak[2]+1]-minval), 
                        log(1+maxval-minval) ); 
end

# 3-point Gaussian subpixel 3D from the voxel neighbourhood around the max peak
function gaussian( corr_mat::Array{T,3}, peak, maxval::T, minval::T=0 ) where {T<:AbstractFloat} 

    return gaussian_3D( log(1+corr_mat[peak[1]-1,peak[2],peak[3]]-minval), 
                        log(1+corr_mat[peak[1]+1,peak[2],peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]-1,peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]+1,peak[3]]-minval), 
                        log(1+corr_mat[peak[1],peak[2],peak[3]-1]-minval), 
                        log(1+corr_mat[peak[1],peak[2],peak[3]+1]-minval),
                        log(1+maxval-minval) ); 
end

function gaussian_2D( up::T, down::T, left::T, right::T, mx::T ) where {T<:AbstractFloat}

    return [ ( up  - down )/( 2* up  - 4*mx + 2*down  ), 
             (left - right)/( 2*left - 4*mx + 2*right ) ];
end

function gaussian_3D( up::T, down::T, left::T, right::T, front::T, back::T, mx::T ) where {T<:AbstractFloat}

    return [ (  up  - down )/( 2*  up  - 4*mx + 2*down  ),
             ( left - right)/( 2* left - 4*mx + 2*right ),
             (front - back )/( 2*front - 4*mx + 2*back  ) ];
end

"""
    Some signal to NOISE implementation
"""

# autocorr
function compute_SN( pivparams, tmp_data, coord_data )
    compute_SN( tmp_data, _smarg( pivparams ), coord_data[5], coord_data[6] )
end
function compute_SN( tmp_data, SM::Dims{N}, SR_TLF_off, SR_BRB_off ) where {N}

    corr = tmp_data[1]; 

    # cross-correlation at the center of the cross-correlation matrix

    cntr = SM .+ 1; 
    mid  = corr[ cntr... ]

    # cross-correlation in the rest of the cross-correlation matrix. 

    # we need to account for the possibility of the search region going of out bounds, 
    # which is what the code below is doing: extracting the coordinates of the 
    # fully overlapping translations that do not involve out-of-bound elements in the
    # search region. 

    trans_TL = ones( Int64, N ) .+ SR_TLF_off
    trans_F  = ( N == 2  ) ? 1 : 1 + SR_TLF_off[3]
    trans_BR = ones( Int64, N ) .* ( 2 .* SM  ) .-  SR_BRB_off; 
    trans_B  = ( N == 2  ) ? 1 : 1 + 2 * SM[3] - SR_BRB_off[3]; 

    TLF  = ( trans_TL..., trans_F )
    BRB  = ( trans_BR..., trans_B )

    fovp = UnitRange.( TLF, BRB ); 
    len  = prod( length.( fovp ) ); 
    avg  = ( sum( corr[ fovp... ] ) - mid )/( len - 1 )

    return mid / avg; 
end
