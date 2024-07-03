################################################################################

function copy_inter_region_minus_mean!( pad_F, F, coord_data::NTuple{N,T} ) where {N,T}
    copy_inter_region_minus_mean!( pad_F, F, coord_data[1], coord_data[2] )
end

function copy_inter_region_minus_mean!( pad_F, F, IR_TLF, IR_BRB )
    pad_F .= 0.0; 
    size_F = IR_BRB .- IR_TLF .+ 1; 
    mean_F = sum( F[ UnitRange.(IR_TLF,IR_BRB)... ] ) / prod( size_F )
    pad_F[ Base.OneTo.(size_F)... ] .= F[ UnitRange.(IR_TLF,IR_BRB)... ] .- mean_F
end

function copy_search_region_minus_mean!( pad_G, G, coord_data::NTuple{N,T} ) where {N,T}
    copy_search_region_minus_mean!( pad_G, G, coord_data[3], coord_data[4], coord_data[5] )
end

function copy_search_region_minus_mean!( pad_G, G, SR_TLF, SR_BRB, SR_TLF_offset )

    pad_G .= 0.0
    size_G = SR_BRB .- SR_TLF .+ 1; 
    mean_G = sum( G[ UnitRange.( SR_TLF, SR_BRB )... ] ) / prod( size_G )
    pad_coords = UnitRange.( 1 .+ SR_TLF_offset, size_G .+ SR_TLF_offset );
    pad_G[ pad_coords... ] .= G[ UnitRange.( SR_TLF, SR_BRB )... ]; 
end

################################################################################

function fftcc2zncc!( pad_F, pad_G, int2_G, sumF2, size_F, size_G  )
    
    # 1-. identifying the range of fully-overlapping translations
    marg_G = div.( size_G .- size_F, 2 ); 
    fovp = UnitRange.( 1, 1 .+ 2 .* marg_G  ); 

    # 2-. pad_F already contains sum(F*G) for all translations, which is the numerator of ZNCC
                                            
    # 3-. Computing denominator within pad_G: sqrt( sum( F ).^2 ) * sqrt( sum( G ).^2 ) 

    # 3.1-. Setting pad_G each pixels to sqrt( sum( G ).^2 ) using integral sums
    pad_G[ fovp... ] .= 0.0
    add_numerator_nsqecc( pad_G, int2_G, size_F, marg_G ) # sum(G²)
    pad_G[ fovp... ] .= sqrt.( pad_G[ fovp... ] )

    # 3.2-. Multiplying pad_G by sqrt( sum( F ).^2 ), which is constant
    pad_G[ fovp... ] .= sqrt( sumF2 )

    # 4-. ZNCC = pad_F ./ pad_G
    pad_F[ fovp... ] ./= pad_G[ fovp... ]; 

    return nothing
end