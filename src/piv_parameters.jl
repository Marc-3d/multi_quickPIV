struct FFTCC       end 
struct ZNCC        end
struct NSQECC      end
struct mask_NSQECC end

CORRTYPES = Union{FFTCC, ZNCC, NSQECC, mask_NSQECC};
dimable   = Union{Integer,Dims{2},Dims{3},Any}; 
toDims3( input::Any     ) = nothing
toDims3( input::Integer ) = ( input, input, input )
toDims3( input::Dims{2} ) = ( input[1], input[2], 0 )
toDims3( input::Dims{3} ) = input

function parseCorr( corr::String )
    if     lowercase(corr) == "zncc";     return   ZNCC();
	elseif lowercase(corr) == "fft";      return  FFTCC();
	elseif lowercase(corr) == "nsqecc";   return NSQECC();
    else;                                 return NSQECC();
	end
end

"""
    Structure that contains all the PIV parameters. The function "setPIVParameters()"
    gives default values to all parameters. 
"""
mutable struct PIVParameters
    corr_alg::CORRTYPES          # cross-correlation algorithm
   interSize::Dims{3}            # interrogation area/volume size in pixels/voxels
searchMargin::Dims{3}            # search margin in pixels/voxels
     overlap::Union{Dims{3}}     # overlap in pixels/voxels between adjacent interrogation regions
        step::Union{Dims{3},Any} # steps between adjacent interrogation regions
   multipass::Integer            # multi-pass depth
   computeSN::Bool               # whether to compute signal-to-noise ratio or not
     filtFun::Function           # Function to filter the background in FTCC, ZNCC and NSQECC
   threshold::Float64            # Threshold for the background filtering function
       ndims::Integer            # Number of dimensions of the data. This is set within PIV(...).
end

# constructor 
function PIVParameters( calg::CORRTYPES, isize::dimable, smarg::dimable, ovp::dimable,
                        step::dimable, mp::Integer, doSN::Bool, filtFun::Function, 
                        th::Real, ndims::Integer )
    return PIVParameters( calg, toDims3(isize), toDims3(smarg), toDims3(ovp), 
                          toDims3(step), mp, doSN, filtFun, Float64(th), ndims )
end

# constructor with default values
function setPIVParameters(; corr_alg = "nsqecc", interSize = 32, searchMargin = 0, 
                            overlap = 0, step = nothing, mpass = 1, computeSN = false, 
                            filtFun = (x)->maxval(x), threshold = -1.0, ndims = 3 )
    return PIVParameters( parseCorr(corr_alg), interSize, searchMargin, overlap,
                          step, mpass, computeSN, filtFun, threshold, ndims )
end

_isize( params::PIVParameters, scale=1 ) = params.interSize[ 1:params.ndims ] .* scale; 
_smarg( params::PIVParameters, scale=1 ) = params.searchMargin[ 1:params.ndims ] .* scale;
_step(  params::PIVParameters, scale=1 ) = ( step == nothing ) ? nothing : params.step[ 1:params.ndims ] .* scale; 
_ovp(   params::PIVParameters, scale=1 ) = params.overlap[1:params.ndims] .* scale; 
_ssize( params::PIVParameters, scale=1 ) = _isize( params, scale ) .+ 2 .* _smarg( params, scale );