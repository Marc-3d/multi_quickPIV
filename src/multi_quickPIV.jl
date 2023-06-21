module multi_quickPIV

include("piv_parameters.jl")
include("./utils/common_ops.jl")
include("./utils/piv_ops.jl")
include("./utils/fft_ops.jl")
include("./utils/nsqecc_ops.jl")
include("./crosscorrelation_algorithms/fftcc.jl")
include("./crosscorrelation_algorithms/zncc.jl")
include("./crosscorrelation_algorithms/nsqecc.jl")
include("./crosscorrelation_algorithms/mask_nsqecc.jl")

function PIV( input1, input2, pivparams::PIVParameters, precision=32 )
	return PIV_singlethreaded( input1, input2, pivparams, precision  )
end

"""
    Single threaded PIV implementation. This is the default. 
"""
function PIV_singlethreaded( input1::AbstractArray{<:Real,N}, input2::AbstractArray{<:Real,N}, 
	                         pivparams::PIVParameters, precision=32 ) where {N}

	size1, size2 = size(input1), size(input2);
    @assert all( size1 .== size2 ) "PIV inputs need to have the same size."

    pivparams.ndims = N;

    # ONE OF {FFT, ZNCC, NSQECC, MASK_NSQECC}
    corr_alg = pivparams.corr_alg; 

    # PREALLOCATING RESULTS: VECTOR FIELD + SIGNAL-TO-NOISE MATRIX
	VF, SN = allocate_outputs( size1, pivparams, precision )
	counts = zeros( UInt16, size( VF )[2:end] );

	# MULTISCALE LOOP
	for scale in pivparams.multipass:-1:1

        # PREALLOCATING CROSS-CORRELATION DATA
        tmp_data = allocate_tmp_data( corr_alg, scale, pivparams, precision )

        # FOR EACH INTERROGATION/SEARCH PAIR IN THE CURRENT SCALE
		vf_size = get_VF_size( size1, pivparams, scale )
        for vf_idx in 1:prod( vf_size )
            
            IA_TLF = get_interrogation_origin( vf_idx, vf_size, pivparams, scale )

            # ONE-LINER TO COMPUTE CROSS-CORRELATION AND FIND MAXIMUM PEAK
            displacement = displacement_from_crosscorrelation( corr_alg, input1, input2, IA_TLF, scale, pivparams, tmp_data )

            # UPDATING THE VECTOR FIELD
            update_vectorfield!( VF, counts, displacement, vf_idx, size1, pivparams, scale )
            
            # SINCE THE CORRELATION MATRIX STILL RESIDES IN TMP_DATA, WE CAN STILL COMPUTE SN FROM IT
            if scale == 1 && pivparams.computeSN
                SN = compute_SN( pivparams, tmp_data )
                SN[vf_idx] = SN;
            end
        end
		
		destroy_fftw_plans( corr_alg, tmp_data )
        #interpolate_vectorfield!( VF, counts ); 
	end


    return VF, SN 
end

"""
    Multithreaded PIV. This is the default if julia is launched with multiple threads. 
"""
function PIV_multithreaded( input1, input2, pivparams, precision=32 )

    @assert size(input1) .== size(input2) "PIV inputs need to have the same size."
    
    corr_alg = pivparams.corr; 
	VF, SN   = allocate_outputs( size(input1), pivparams, precision=precision)

	for scale in pivparams.multipass:-1:1 
		
		tmp_data = [ allocate_tmp_data( corr_alg, scale, pivparams, precision ) for tidx in nthreads ]
        
		#@threads ... 
			# EACH THREAD/ PROCESSES AN INTERROGATION/SEARCH REGION
			IA_start  = get_interrogation_origin( idx, size(input1), s )

			# ONE-LINER TO COMPUTE CROSS-CORRELATION AND FIND MAXIMUM PEAK
            displ, SN = crosscorrelate_and_return_displacement( corr_alg, IA_start, mp, pivparams, tmp_data[tidx] )
			
			# UPDATING THE VECTOR FIELD
			# vf_coords = ... 
			update_vectorfield!( U, V, W, SN, vf_coords, displ )
		#end

		# SYNCH THREADS
	end

end

"""
    TODO: CPU(single_thread)s-GPU PIV. 
"""
function PIV_singlethreaded_and_GPU( input1, input2, pivparams; precision=32 )

    @assert size(input1) .== size(input2) "PIV inputs need to have the same size."
    
    corr_alg = pivparams.corr; 

    # PREALLOCATING RESULTS: VECTOR FIELD + SIGNAL-TO-NOISE MATRIX
	VF, SN = allocate_outputs( size(input1), pivparams, precision=precision)

	# MULTISCALE LOOP
	for scale in pivparams.multipass:-1:1 
		
        # ALLOCATING DATA FOR EACH THREAD/STREAM
		tmp_data = [ allocate_tmp_data( corr_alg, scale, pivparams, precision=precision ) for tidx in nthreads ]
        
		#@threads ... 
			# EACH THREAD/STREAM PROCESSES AN INTERROGATION/SEARCH REGION
			IA_start  = get_interrogation_origin( idx, size(input1), s )

			# ONE-LINER TO COMPUTE CROSS-CORRELATION AND FIND MAXIMUM PEAK
            displ, SN = crosscorrelate_and_return_displacement( corr_alg, IA_start, mp, pivparams, tmp_data[tidx] )
			
			# UPDATING THE VECTOR FIELD
			# vf_coords = ... 
			update_vectorfield!( U, V, W, SN, vf_coords, displ )
		#end

		# SYNCH THREADS
	end

end


"""
    TODO: CPU(multi_thread)s-GPU PIV. 
"""
function PIV_multithreaded_and_GPU( input1, input2, pivparams; precision=32 )

    @assert size(input1) .== size(input2) "PIV inputs need to have the same size."
    
    corr_alg = pivparams.corr; 

    # PREALLOCATING RESULTS: VECTOR FIELD + SIGNAL-TO-NOISE MATRIX
	VF, SN = allocate_outputs( size(input1), pivparams, precision=precision)

	# MULTISCALE LOOP
	for scale in pivparams.multipass:-1:1 
		
        # ALLOCATING DATA FOR EACH THREAD/STREAM
		tmp_data = [ allocate_tmp_data( corr_alg, scale, pivparams, precision=precision ) for tidx in nthreads ]
        
		#@threads ... 
			# EACH THREAD/STREAM PROCESSES AN INTERROGATION/SEARCH REGION
			IA_start  = get_interrogation_origin( idx, size(input1), s )

			# ONE-LINER TO COMPUTE CROSS-CORRELATION AND FIND MAXIMUM PEAK
            displ, SN = crosscorrelate_and_return_displacement( corr_alg, IA_start, mp, pivparams, tmp_data[tidx] )
			
			# UPDATING THE VECTOR FIELD
			# vf_coords = ... 
			update_vectorfield!( VF, SN, vf_coords, displ )
		#end

		# SYNCH THREADS
	end

end



end 
