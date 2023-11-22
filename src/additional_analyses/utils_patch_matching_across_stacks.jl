"""
    RETRUNS A TUPLE WITH ALL COORDINATE INFORMATION AOUT THE INTERROGATION AND SEARCH REGIONS.
"""
function get_interr_and_search_coordinates( vf_idx::Int, 
                                            vf_size::Dims{N}, 
                                            input_size::Dims{N}, 
                                            offset,
                                            scale, 
                                            pivparams::PIVParameters
                                        ) where {N}

    IR_TLF, IR_BRB, IR_TLF_off, IR_BRB_off = get_interr_coordinates( vf_idx, vf_size, input_size, offset, scale, pivparams )
    SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off = get_search_coordinates( vf_idx, vf_size, input_size, offset, scale, pivparams )

    # Adding IR_TLF_off and IR_BRB_off at the end, for compatibility reasons
    return ( IR_TLF, IR_BRB, SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off, IR_TLF_off, IR_BRB_off )
end


function get_interr_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N},
                                 offset, scale, pivparams::PIVParameters ) where {N}

    # Computing interrogation limits as usual
    vf_coords = linear2cartesian( vf_idx, vf_size ); 
    IR_TLF    = ones(Int, N) .+ ( vf_coords .- 1 ) .* _step( pivparams, scale );
    IR_BRB    = IR_TLF .+ _isize( pivparams, scale ) .- 1; 

    # Applying the offset to interrogation limits
    IR_TLF = IR_TLF .+ round.( Int, offset ); 
    IR_BRB = IR_BRB .+ round.( Int, offset );

    # After the offset is applied, the interrogation limits may go out of bounds...
    IR_TLF_off = abs.( min.( 0, IR_TLF .+ offset .- 1 ) );
    IR_BRB_off = abs.( min.( 0, input_size .- ( IR_BRB .+ offset ) ) );
    IR_TLF = max.(      1    , IR_TLF ); 
    IR_BRB = min.( input_size, IR_BRB ); 

    return IR_TLF, IR_BRB, IR_TLF_off, IR_BRB_off
end


function get_search_coordinates( vf_idx::Int, vf_size::Dims{N}, input_size::Dims{N},
                                 offset, scale, pivparams::PIVParameters ) where {N}

    vf_coords  = linear2cartesian( vf_idx, vf_size ); 

    # Interrogation coordinates with offset
    IR_TLF = ones(Int, N) .+ ( vf_coords .- 1 ) .*  _step( pivparams, scale );
    IR_BRB = IR_TLF .+  _isize( pivparams, scale ) .- 1; 
    IR_TLF = IR_TLF .+ round.( Int, offset ); 
    IR_BRB = IR_BRB .+ round.( Int, offset );

    # Search coordinates, correcting for possible out-of-bounds
    SR_TLF = max.( 1, IR_TLF .- _smarg( pivparams, scale ) ); 
    SR_BRB = min.( input_size, IR_BRB .+ _smarg( pivparams, scale ) );

    # Possible out-of-bounds extent
    SR_TLF_off = abs.( min.( 0, IR_TLF .- _smarg( pivparams, scale ) .- 1 ) );
    SR_BRB_off = abs.( min.( 0, input_size .- ( IR_BRB .+ _smarg( pivparams,scale) ) ) ); 

    return SR_TLF, SR_BRB, SR_TLF_off, SR_BRB_off
end