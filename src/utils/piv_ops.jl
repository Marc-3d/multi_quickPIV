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
    Allocates output matrices to hold the PIV vector field (VF) and the signal-to-noise (SN) 
    matrix (if desired). 2D VFs are stored in a 3D matrix of size (3,VF_heigh,VF_width), and
    3D VFs in a 4D matrice of size (3,VF_heigh,VF_width,VF_depth). 
"""
function allocate_outputs( input_size::Dims{N}, pivparams::PIVParameters, precision=32 ) where {N}

    VFsize = get_vectorfield_size( input_size, pivparams, 1 );
    VFtype = ( precision == 32 ) ? Float32 : Float64; 
    VF     = zeros( VFtype, N, VFsize... );
    SN     = ( pivparams.computeSN ) ? zeros( VFtype, VFsize... ) : nothing;
    return VF, SN
end


"""
    Finding the TLF (top-left-front) and BRB (bottom-right-back) coordinates of the "i"th 
    interrogation region.
"""
function get_interrogation_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N}, scale, pivparams::PIVParameters ) where {N}

    vf_coords = linear2cartesian( vf_idx, vf_size ); 
    IR_TLF    = ones(Int, N) .+ ( vf_coords .- 1 ) .* _step( pivparams, scale );
    IR_BRB    = IR_TLF .+ _isize( pivparams, scale ) .- 1; 

    # if we are using convolution to replace cross-correlation, the interrogation input is reversed
    if eltype( pivparams.corr_alg ) <: CONVTYPES
        IR_TLF = input_size .- IR_BRB .+ 1; 
        IR_BRB = input_size .- IR_TLF .+ 1; 
    end
    return IR_TLF, IR_BRB
end


"""
    Finding the TLF (top-left-front) and BRB (bottom-right-back) coordinates of the "i"th 
    search region.
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
    Return a tuple with all coordinate information about the interrogation and search regions
"""
function get_interrogation_and_search_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N}, scale, pivparams::PIVParameters ) where {N}

    IR_TLF, IR_BRB = get_interrogation_coordinates( vf_idx, vf_size, input_size, scale, pivparams )
    SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off = get_search_coordinates( vf_idx, vf_size, input_size, scale, pivparams )

    return ( IR_TLF, IR_BRB, SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off )
end


"""
    USED FOR COPYING THE INPUT DATA INTO THE PADDED ARRAYS FOR FFT
"""

# The two functions below accept a tuple of coordinates, which makes the code more succint.
# "coord_data" contains = [ IA_TLF, IA_BRB, SA_TLF, SA_BRB, SA_TLF_off, SA_BRB_off ].

function copy_inter_region!( pad_inter, input1, coord_data::NTuple{6,N} ) where {N}
    copy_inter_region!( pad_inter, input1, coord_data[1], coord_data[2] )
end

function copy_search_region!( pad_inter, input1, coord_data::NTuple{6,N} ) where {N}
    copy_search_region!( pad_inter, input1, coord_data[3], coord_data[4], coord_data[5] )
end

# 

function copy_inter_region!( pad_inter, input1, IR_TLF, IR_BRB )
    pad_inter .= 0.0; 
    inter_size = IR_BRB .- IR_TLF .+ 1; 
    pad_inter[ Base.OneTo.(inter_size)... ] .= input1[ UnitRange.(IR_TLF,IR_BRB)... ]
end

function copy_search_region!( pad_search, input2, SR_TLF, SR_BRB, SR_TLF_offset )

    pad_search .= 0.0
    search_size = SR_BRB .- SR_TLF .+ 1; 
    padsearch_coords = UnitRange.( 1 .+ SR_TLF_offset,  search_size .+ SR_TLF_offset );
    pad_search[ padsearch_coords... ] .= input2[ UnitRange.( SR_TLF, SR_BRB )... ]; 
end


"""
    USED MOSTELY FOR FILTERING INTERROGATION REGIONS THAT BELONG TO THE BACKGROUND. 
    GENERALLY, THOSE WITH A VERY LOW MEAN INTENSITY CORRESPOND TO BACKGROUND REGIONS.
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
    
"""

"""
    Getting vector field coordinates for each displacement vector would be 
    quite trivial... it is the linear index "VFidx" of the interrogatoin /
    search pair in multi_quickPIV.jl:29. 
    
    However, we need to account for multipass, where displacements computed
    at high scales (4x, 3x, etc) are used to offset the sampling of search 
    regions in the next iterations (at lower scales).

    For example, a single 2x interrogation/search pair contains 4 (2x2) 
    1x interrogation/search pairs. The corresponding 2x displacement will be 
    used to offset the search regions of the 4 contained 1x interrogation / 
    search pairs. 

    In addition, overlap between interrogation areas requires interpolation
    of the vectors computed at large scales. 
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
function get_vf_coords( inter_index, input_size::Dims{N}, pivparams::PIVParameters, scale ) where {N}

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

function gaussian_displacement( corr_mat, scale, pivparams::PIVParameters )

    # cross-correlation maximum peak coordinates 
    peak, maxval = firstPeak( corr_mat )

    # unpadded cross-correlation center is very simple, searchMargin .+ 1. 
    center = _smarg( pivparams, scale ) .+ 1; 

    displacement = gaussian_refinement( corr_mat, peak, maxval ) .- center

    return displacement
end

function gaussian_displacement( corr_mat, isize::Dims{N}, ssize::Dims{N} ) where {N}

    peak, maxval = firstPeak( corr_mat )
    center = div.( isize, 2 ) .+ div.( ssize, 2 )
    return gaussian_refinement( corr_mat, peak, maxval ) .- center
end

function gaussian_refinement( corr_mat::Array{T,N}, peak, maxval ) where {T,N}

    if all( peak .> 1 ) && all( peak .< size(corr_mat) )
        minval = minimum( corr_mat[ UnitRange.( peak .- ones(Int,N), peak .+ ones(Int,N) )... ] )
        gaussi = gaussian( corr_mat, peak, maxval, T(minval) )
        return peak .+ gaussi
    else
        return peak 
    end
end

# 3-point Gaussian subpixel 2D from the pixel neighbourhood around the max peak 
function gaussian( corr_mat::Array{T,2}, peak, maxval::T, minval::T=0 ) where {T}

    return gaussian_2D( log(1+corr_mat[peak[1]+1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1]-1,peak[2]]-minval), 
                        log(1+corr_mat[peak[1],peak[2]-1]-minval), 
                        log(1+corr_mat[peak[1],peak[2]+1]-minval), 
                        log(1+maxval-minval) ); 
end

# 3-point Gaussian subpixel 3D from the voxel neighbourhood around the max peak
function gaussian( corr_mat::Array{T,3}, peak, maxval::T, minval::T=0 ) where {T<:AbstractFloat} 

    return gaussian_3D( log(1+corr_mat[peak[1]+1,peak[2],peak[3]]-minval), 
                        log(1+corr_mat[peak[1]-1,peak[2],peak[3]]-minval), 
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