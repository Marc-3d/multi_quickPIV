function fftcc2nsqecc!( pad_F, pad_G, int2_G, sumF2, size_F, size_G  )
    
    # 1-. identifying the range of fully-overlapping translations
    marg_G = div.( size_G .- size_F, 2 ); 
    fovp = UnitRange.( 1, 1 .+ 2 .* marg_G  ); 

    # 2-. Manipulating pad_F, which initially contains sum(G*F) for all translations
    pad_F[ fovp... ] .*= -2;                              # - 2sum(G*F)
    pad_F[ fovp... ] .+= sumF2;                           # sum(F²) - 2sum(G*F)
    add_numerator_nsqecc( pad_F, int2_G, size_F, marg_G ) # sum(F²) + sum(G²) - 2sum(G*F)
                                            
    # 3.2-. Computing denominator within pad_G. 
    pad_G[ fovp... ]  .= 0.0
    add_numerator_nsqecc( pad_G, int2_G, size_F, marg_G ) # sum(G²)
    pad_G[ fovp... ]  .= sqrt.( pad_G[ fovp... ] )        # √sum(G²)
    pad_G[ fovp... ] .*= sqrt( sumF2 )                    # √sum(F²)√sum(G²)

    # 3.3-. numerator / denominator within pad_F
    # pad_F[ fovp... ] ./= pad_G[ fovp... ];

    # 3.3-. 1/( 1 + num/den ) = 1/( (den + num)/den ) = den/(den+num)
    pad_F[ fovp... ] .+= pad_G[ fovp... ]; 
    pad_F[ fovp... ]  .= pad_G[ fovp... ] ./ pad_F[ fovp... ]; 

    return nothing
end