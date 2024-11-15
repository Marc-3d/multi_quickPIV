module multi_quickPIV

# YOUTUBE VIDEO IF YOU WANT A DEEP DIVE INTO CROSS-CORRELATION AND HOW IT IS IMPLEMENTED IN quickPIV.
# OR README in the ./crosscorrelation_algorithms. 
# TODO: record and upload video

include("piv_parameters.jl")
include("./utils/common_ops.jl")	
include("./utils/piv_ops.jl")
include("./utils/fft_ops.jl")
include("./utils/nsqecc_ops.jl")
include("./utils/masked_nsqecc_ops.jl")
include("./crosscorrelation_algorithms/fftcc.jl")
include("./crosscorrelation_algorithms/zncc.jl")
include("./crosscorrelation_algorithms/nsqecc.jl")
include("./crosscorrelation_algorithms/mask_nsqecc.jl")
include("./postprocessing/postprocessing.jl")
#include("./convolution_algorithms/fftc.jl")

# STANDARD PIV 
function PIV( input1, input2, pivparams::PIVParameters=setPIVParameters(); precision=64 )
	return PIV_CPU( input1, input2, pivparams, precision  )
end

# MASKED PIV
function PIV( input1, input2, mask, pivparams::PIVParameters=setPIVParameters(); precision=64 )
	return PIV_CPU_masked( input1, input2, mask, pivparams, precision=precision  )
end


"""
	CPU PIV implementation.

	If julia is launched with multiple threads, we enable FFTW to perform multithreaded r2c/c2r FFT 
	transfroms. For instance, to launch a julia REPL with 4 threads run: "julia -t 4". If you want 
	to run multithreaded quickPIV from a jupyter notebook, follow the instructions in this github 
	issue: https://github.com/JuliaLang/IJulia.jl/issues/882#issuecomment-579520246 
"""
function PIV_CPU( input1::AbstractArray{<:Real,N}, 
	              input2::AbstractArray{<:Real,N}, 
	              pivparams::PIVParameters, 
				  precision=32
				) where {N}

	size1, size2 = size(input1), size(input2);
    @assert all( size1 .== size2 ) "PIV inputs need to have the same size."

    pivparams.ndims = N;

    # PREALLOCATING RESULTS: VECTOR FIELD + SIGNAL-TO-NOISE MATRIX
	VF, SN = allocate_outputs( size1, pivparams, precision )

	# MULTISCALE LOOP
	for scale in pivparams.multipass:-1:1

        # PREALLOCATING CROSS-CORRELATION DATA
        tmp_data = allocate_tmp_data( pivparams.corr_alg, scale, pivparams, precision )
		vf_size  = get_vectorfield_size( size1, pivparams, scale )
		counts   = zeros( UInt16, vf_size );

        for vf_idx in 1:prod( vf_size )
            
        	# COMPUTING COORDINATES FOR THE CURRENT PAIR OF INTERROGATION/SEARCH REGION
			coord_data = get_interrogation_and_search_coordinates( vf_idx, vf_size, size1, scale, pivparams ); 

			# (OPTIONAL) FILTERING OF INTERROGATION REGIONS IF pivparams.filtFun(IR) < pivparams.threshold
			skip_inter_region( input1, coord_data[1], coord_data[2], pivparams ) && ( continue; )

			# COPYING INTERROGATION/SEARCH REGIONS INTO PADDED ARRAYS FOR FFT
			prepare_inputs!( pivparams.corr_alg, input1, input2, coord_data, tmp_data );

			# COMPUTING CROSS-CORRELATION MATRIX
			crosscorrelation!( pivparams.corr_alg, scale, pivparams, tmp_data ); 

			# COMPUTING DISPLACEMENT FROM MAXIMUM PEAK OF CC + GAUSSIAN REFINEMENT
            displacement = gaussian_displacement( tmp_data[1], scale, pivparams, coord_data, tmp_data )

            # UPDATING THE VECTOR FIELD
            update_vectorfield!( VF, counts, displacement, vf_idx, size1, pivparams, scale )
            
            # (OPTIONAL) COMPUTING SIGNAL-TO-NOISE RATIO
            if scale == 1 && pivparams.computeSN
                SN[vf_idx] = compute_SN( pivparams, tmp_data, coord_data );
            end
        end
		
		destroy_fftw_plans( pivparams.corr_alg, tmp_data )
        #interpolate_vectorfield!( VF, counts ); 
	end

    return VF, SN 
end

# MASKED PIV
function PIV_CPU_masked( input1::AbstractArray{<:Real,N}, 
						 input2::AbstractArray{<:Real,N}, 
						 mask::AbstractArray{<:Real,N}, 
						 pivparams::PIVParameters; 
						 precision=32
					   ) where {N}

	size1, size2, size3 = size(input1), size(input2), size( mask );
    @assert all( size1 .== size2 .== size3 ) "PIV inputs need to have the same size."
    
	pivparams.ndims = N;
	corr_alg = mask_NSQECC()

	VF, SN = allocate_outputs( size1, pivparams, precision )

	for scale in pivparams.multipass:-1:1

        tmp_data = allocate_tmp_data( corr_alg, scale, pivparams, precision )
		vf_size  = get_vectorfield_size( size1, pivparams, scale )
		counts   = zeros( UInt16, vf_size );

        for vf_idx in 1:prod( vf_size )
            
        	# COMPUTING COORDINATES FOR THE CURRENT PAIR OF INTERROGATION/SEARCH REGION
			coord_data = get_interrogation_and_search_coordinates( vf_idx, vf_size, size1, scale, pivparams ); 

			# (OPTIONAL) FILTERING OF INTERROGATION REGIONS IF pivparams.filtFun(IR) < pivparams.threshold
			skip_inter_region( input1, mask, coord_data[1], coord_data[2], pivparams ) && ( continue; )

			# COPYING INTERROGATION/SEARCH REGIONS INTO PADDED ARRAYS FOR FFT
			prepare_inputs!( corr_alg, input1, input2, mask, coord_data, tmp_data );

			crosscorrelation!( corr_alg, scale, pivparams, tmp_data ); 

            displacement = gaussian_displacement( tmp_data[1], scale, pivparams, coord_data, tmp_data )

            # UPDATING THE VECTOR FIELD
            update_vectorfield!( VF, counts, displacement, vf_idx, size1, pivparams, scale )
            
            # (OPTIONAL) COMPUTING SIGNAL-TO-NOISE RATIO
            if scale == 1 && pivparams.computeSN
                SN[vf_idx] = compute_SN( pivparams, tmp_data );
            end
			
			#println( displacement )
			#return VF, tmp_data[1]
        end
		
		destroy_fftw_plans( corr_alg, tmp_data )
        #interpolate_vectorfield!( VF, counts ); 
	end


    return VF, SN 
end

"""
    TODO: CPU(single_thread)s-GPU PIV. 
"""
function PIV_CPUGPU( input1, input2, pivparams; precision=32 )

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

include("./additional_analyses/utils_patch_matching_across_stacks.jl")
include("./additional_analyses/patch_matching_across_stack.jl")

end 