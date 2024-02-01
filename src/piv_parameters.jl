# cross-correlation (CC) algorithms
struct FFTCC       end 
struct ZNCC        end
struct NSQECC      end
struct mask_NSQECC end

# convolution (C) can replace cross-correlation if we mirror the first input.
struct FFTC       end 
struct ZNC        end
struct NSQEC      end
struct mask_NSQEC end


CONVTYPES = Union{FFTC, ZNC, NSQEC, mask_NSQEC}
CORRTYPES = Union{FFTCC, ZNCC, NSQECC, mask_NSQECC,FFTC, ZNC, NSQEC, mask_NSQEC};
dimable   = Union{Integer,Dims{2},Dims{3},Any}; 
toDims3( input::Any     ) = nothing
toDims3( input::Integer ) = ( input, input, input )
toDims3( input::Dims{2} ) = ( input[1], input[2], 0 )
toDims3( input::Dims{3} ) = input

function parseCorr( corr::String )
    if     lowercase(corr) == "zncc";     return   ZNCC();
	elseif lowercase(corr) == "fft";      return  FFTCC();
	elseif lowercase(corr) == "nsqecc";   return NSQECC();
    elseif lowercase(corr) == "znc";      return    ZNC();
    elseif lowercase(corr) == "fftc";     return   FFTC();
    elseif lowercase(corr) == "nsqec";    return  NSQEC();
    else;                                 return NSQECC();
	end
end

"""
    Structure that contains all the PIV parameters. The function "setPIVParameters()"
    gives default values to all parameters. 
"""
mutable struct PIVParameters
    corr_alg::CORRTYPES          # Cross-correlation algorithm
    unpadded::Bool               # Whether to remove padding to the minimum to speed up FFTs
    good_pad::Bool               # Whether to add padding to ease factorization and speed up FFTs
     odd_pad::Bool               # Whether to add padding to turn odd FFTs sizes to even FFT sizes.
   interSize::Dims{3}            # Interrogation area/volume size in pixels/voxels
searchMargin::Dims{3}            # Search margin in pixels/voxels
     overlap::Union{Dims{3}}     # Overlap in pixels/voxels between adjacent interrogation regions
        step::Union{Dims{3},Any} # Steps between adjacent interrogation regions
   multipass::Integer            # Multi-pass depth
   computeSN::Bool               # Whether to compute signal-to-noise ratio or not
     filtFun::Function           # Function to filter the background in FTCC, ZNCC and NSQECC
   threshold::Float64            # Threshold for the background filtering function
       ndims::Integer            # Number of dimensions of the data. This is set within PIV(...).
      ovp_th::Float64            # For masked_NSQECC
end

# constructor 
function PIVParameters( calg::CORRTYPES, 
                        unpadded::Bool, good_pad::Bool, odd_pad::Bool,
                        isize::dimable, smarg::dimable, ovp::dimable, step::dimable, 
                        multipass::Integer, computeSN::Bool, 
                        filtFun::Function, threshold::Real, 
                        ndims::Integer, 
                        ovp_th::Float64
                    )
    return PIVParameters( calg, 
                          good_pad, unpadded, odd_pad,
                          toDims3(isize), toDims3(smarg), toDims3(ovp), toDims3(step), 
                          multipass, computeSN, 
                          filtFun, Float64(threshold), 
                          ndims, ovp_th )
end

# constructor with default values
function setPIVParameters(; corr_alg = "nsqecc", 
                            unpadded = true, good_pad = true, odd_pad = false, 
                            interSize = 32, searchMargin = 10,  overlap = 0, step = nothing, 
                            multipass = 1, computeSN = false, 
                            filtFun = (x)->maxval(x), threshold = -1.0, 
                            ndims = 3, ovp_th = 0.5 )
    return PIVParameters( parseCorr(corr_alg), 
                          unpadded, good_pad, odd_pad, 
                          interSize, searchMargin, overlap, step,
                          multipass, computeSN, 
                          filtFun, threshold, 
                          ndims, ovp_th )
end

_ovp(   params::PIVParameters, scale=1 ) = params.overlap[ 1:params.ndims ] .* scale; 
_isize( params::PIVParameters, scale=1 ) = params.interSize[ 1:params.ndims ] .* scale; 
_smarg( params::PIVParameters, scale=1 ) = params.searchMargin[ 1:params.ndims ] .* scale;
_ssize( params::PIVParameters, scale=1 ) = _isize( params, scale ) .+ 2 .* _smarg( params, scale );
_step(  params::PIVParameters, scale=1 ) = ( params.step === nothing ) ? _isize( params, scale ) .- _ovp( params, scale ) : params.step[ 1:params.ndims ] .* scale; 
