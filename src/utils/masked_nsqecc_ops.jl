"""
    SPECIALIZED FUNCTIONS TO COPY DATA INTO PADDED ARRAYS FOR MASKED_NSQECC
"""

function copy_inter_masked!( pad_arr, arr, mask, coord_data::NTuple{6,N} ) where {N}
    copy_inter_masked!( pad_arr, arr, mask, coord_data[1], coord_data[2] )
end

function copy_inter_squared!( pad_arr, arr, coord_data::NTuple{6,N} ) where {N}
    copy_inter_squared!( pad_arr, arr, coord_data[1], coord_data[2] )
end

function copy_search_squared!( pad_arr, arr, coord_data::NTuple{6,N} ) where {N}
    copy_search_squared!( pad_arr, arr, coord_data[3], coord_data[4], coord_data[5] )
end

# 

function copy_inter_masked!( pad_inter, input1, mask, IR_TLF, IR_BRB )

    pad_inter .= 0.0
    inter_size = IR_BRB .- IR_TLF .+ 1; 
    @inbounds pad_inter[ Base.OneTo.( inter_size )... ]  .= input1[ UnitRange.( IR_TLF, IR_BRB )... ]
    @inbounds pad_inter[ Base.OneTo.( inter_size )... ] .*=   mask[ UnitRange.( IR_TLF, IR_BRB )... ]
end

function copy_inter_squared!( pad_inter, input1, IR_TLF, IR_BRB )

    pad_inter .= 0.0; 
    inter_size = IR_BRB .- IR_TLF .+ 1; 
    @inbounds pad_inter[ Base.OneTo.( inter_size )... ]  .= input1[ UnitRange.(IR_TLF,IR_BRB)... ]
    @inbounds pad_inter[ Base.OneTo.( inter_size )... ] .*= input1[ UnitRange.(IR_TLF,IR_BRB)... ]
end

function copy_search_squared!( pad_search, input2, SR_TLF, SR_BRB, SR_TLF_offset )

    pad_search .= 0.0
    search_size = SR_BRB .- SR_TLF .+ 1; 
    padsearch_coords = UnitRange.( 1 .+ SR_TLF_offset,  search_size .+ SR_TLF_offset );
    @inbounds pad_search[ padsearch_coords... ]  .= input2[ UnitRange.( SR_TLF, SR_BRB )... ]; 
    @inbounds pad_search[ padsearch_coords... ] .*= input2[ UnitRange.( SR_TLF, SR_BRB )... ]; 
end

# INTEGRAL ARRAY IN-PLACE CONSTRUCTION

function integralArray!( int_array::Array{<:AbstractFloat,N}, array::Array{<:Real,N} ) where {N}
    integralArray!( int_array, array, Tuple( ones(N) ), size( array ) )
end

function integralArray!( int_array::Array{<:AbstractFloat,2}, array::Array{<:Real,2}, TL=(1,1), inp_size=size(array) )
    TL   = TL .- 1; 
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        int_array[r,c] = array[TL[1]+r-1,TL[2]+c-1] + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end
end

function integralArray!( int_array::Array{<:AbstractFloat,3}, array::Array{<:Real,3}, TLF=(1,1,1), inp_size=size(array) )
    TLF     = TLF .- 1; 
    h, w, d = inp_size;
    @inbounds for z in 1+1:d+1
        for c in 1+1:w+1
            tmp = 0.0; 
            for r in 1+1:h+1      
                val2 = array[TLF[1]+r-1,TLF[2]+c-1,TLF[3]+z-1]
                int_array[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
                tmp += val2; 
            end 
        end
    end
end


# this one accepts a mask for masked_NSQECC

function integralArraySQ!( int_array::Array{T,N}, 
                           array::Array{<:Real,N},
                           mask::Array{<:Real,N}
                         ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, mask, Tuple(ones(N)), size(array) )

    return nothing
end

function integralArraySQ!( int_array::Array{T,2}, 
                           array::Array{<:Real,2},
                           mask::Array{<:Real,2}, 
                           TL=(1,1), 
                           inp_size=size(array)
                         ) where {T<:AbstractFloat}
    TL   = TL .- 1; 
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        val2 = T(array[TL[1]+r-1,TL[2]+c-1]*mask[TL[1]+r-1,TL[2]+c-1])^2; 
        int_array[r,c] = val2 + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end

    return nothing
end

function integralArraySQ!( int_array::Array{T,3},
                           array::Array{<:Real,3},
                           mask::Array{<:Real,3},
                           TLF=(1,1,1),
                           inp_size=size(array)
                         ) where {T<:AbstractFloat}
    TLF     = TLF .- 1; 
    h, w, d = inp_size;
    for z in 1+1:d+1
        for c in 1+1:w+1
            tmp = 0.0; 
            for r in 1+1:h+1      
                val2 = T(array[TLF[1]+r-1,TLF[2]+c-1,TLF[3]+z-1]*mask[TLF[1]+r-1,TLF[2]+c-1,TLF[3]+z-1])^2
                int_array[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
                tmp += val2; 
            end 
        end
    end

    return nothing
end

function fftcc2masknsqecc!( pad_maskF, pad_G2, sumF2, size_F, size_G  )

    # 1-. identifying the range of fully-overlapping translations
    marg_G = div.( size_G .- size_F, 2 ); 
    fovp = UnitRange.( 1, 2 .* marg_G  ); 

    # 2-. pad_maskF : sum( mask_F * G ) -> sum(F²) + sum(G²) - 2sum(mask_F*G)
    pad_maskF[ fovp... ] .*= -2;                  # - 2sum(mask_F*G)
    pad_maskF[ fovp... ] .+= sumF2;               # sum(F²) - 2sum(mask_F*G)
    pad_maskF[ fovp... ] .+= pad_G2[ fovp... ];   # sum(F²) + sum(mask_G²) - 2sum(mask_F*G)
                                            
    # 3-. pad_G2: sum(mask_G²) -> √sum(mask_F²)√sum(mask_G²)
    pad_G2[ fovp... ] .= sqrt.( abs.( pad_G2[ fovp... ] ) )   # √sum(G²)
    pad_G2[ fovp... ] .*= sqrt( sumF2 )                       # √sum(F²)√sum(G²)

    # 3.3-. ---> 1/( 1 + num/den ) = 1/( (den + num)/den ) = den/(den+num)
    pad_maskF[ fovp... ] .+= pad_G2[ fovp... ];                          # den + num
    pad_maskF[ fovp... ] .=  pad_G2[ fovp... ] ./ pad_maskF[ fovp... ] ; # den/(den+num)

    return nothing
end