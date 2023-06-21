# PIV operations related to creating and managing the output vector fields

"""
    Find out VF size ( number of interrogation areas that fit in each dimension ).
    The VF size can be used to compute the coordinates of each interrogation area
    within the vector field. The scale parameter allows to account for multipass. 

    NOTE: instead of "pivparams.interSize", we use "_isize(pivparams)" because the
    latter will return "pivparams.interSize[1:pivparams.ndims]", making sure that
    isize has the same dimenisons as input_size. Same goes for "step" and "ovp".
"""
function get_VF_size( input_size::Dims{N}, pivparams::PIVParameters, scale=1 ) where {N}
    isize = _isize( pivparams);
    step  = (pivparams.step == nothing ) ? isize .- _ovp(pivparams) : _step(pivparams); 
    get_VF_size( input_size, isize, step, scale )
end

function get_VF_size( input_size::Dims{N}, isize::Dims{N}, step::Dims{N}, scale=1 ) where {N}
    return length.( StepRange.( 1, step .* scale, input_size .- isize .* scale ) );
end

"""
    Allocates the matrices to hold the output vector field: VF = [ U ;; V ] in 2D
    and VF = [ U ;;; V ;;; W ] in 3D. In addition, we can allocate memory for the
    signal-to-noise matrix, if we so desire.
"""
function allocate_outputs( input_size::Dims{N}, pivparams, precision=32 ) where {N}

    VFsize = get_VF_size( input_size, pivparams, 1 );
    VFtype = ( precision == 32 ) ? Float32 : Float64; 
    VF = zeros( VFtype, N, VFsize... ); #Array{VFtype, N+1}( undef, N, VFsize... ); 
    SN = ( pivparams.computeSN ) ? zeros( VFtype, VFsize... ) : nothing;

    return VF, SN
end

"""
    Finding the TLF coordinates of the "i"th interrogation region.
"""
function get_interrogation_origin( vf_idx, vf_size::Dims{N}, scale, pivparams ) where {N}

    vf_coords = linear2cartesian_2( vf_idx, size2strides(vf_size) ); 
    isize     = _isize(pivparams, scale);
    step      = (pivparams.step == nothing ) ? isize .- _ovp(pivparams, scale) : _step(pivparams, scale); 
    TLF       = ones(Int, N) .+ ( vf_coords .- 1 ) .* step
    return Tuple( TLF )
end

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
function get_vf_coords( inter_index, input_size::Dims{N}, pivparams, scale ) where {N}

    if scale == 1
        VFsize    = get_VF_size( input_size, pivparams, scale );
        VFcoords  = linear2cartesian_2( inter_index, size2strides(VFsize) ); 
        return VFcoords
    end

    # VFsize, isize and istep at the lowest scale
    VFsize_1x = get_VF_size( input_size, pivparams, 1 );
    isize_1x = _isize( pivparams );
    istep_1x = (pivparams.step == nothing ) ? isize_1x .- _ovp(pivparams) : _step(pivparams); 

    # VF cartesian coordinates, isize and istep from current scale
    VFsize    = get_VF_size( input_size, pivparams, scale );
    VFcoords  = linear2cartesian_2( inter_index, size2strides( VFsize ) ); 
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
function update_vectorfield!( VF, counts, displacement, vf_idx, input_size::Dims{N}, pivparams, scale ) where {N}

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