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
                val2          = array[TLF[1]r-1,TLF[1]c-1,TLF[1]z-1]
                intArr[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
                tmp          += val2; 
            end 
        end
    end
end

# SQUARE THE INPUT WHILE IN-PLACE CONSTRUCTION OF THE INTEGRAL ARRAY

# implementation without a mask for NSQECC

function integralArraySQ!( int_array::Array{T,N}, array::Array{<:Real,N} ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, Tuple( ones(N) ), size( array ) )
end

function integralArraySQ!( int_array::Array{T,2}, array::Array{<:Real,2}, TL=(1,1), inp_size=size(array) ) where {T<:AbstractFloat}
    TL   = TL .- 1; 
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        int_array[r,c] = T(array[TL[1]+r-1,TL[2]+c-1])^2 + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end
end

function integralArraySQ!( int_array::Array{T,3}, array::Array{<:Real,3}, TLF=(1,1,1), inp_size=size(array) ) where {T<:AbstractFloat}
    TLF     = TLF .- 1; 
    h, w, d = inp_size;
    @inbounds for z in 1+1:d+1
        for c in 1+1:w+1
            tmp = 0.0; 
            for r in 1+1:h+1      
                val2          = T(array[TLF[1]r-1,TLF[1]c-1,TLF[1]z-1])^2
                intArr[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
                tmp          += val2; 
            end 
        end
    end
end

# this one accepts a mask for masked_NSQECC

function integralArraySQ!( int_array::Array{T,N}, array::Array{<:Real,N}, mask::Array{<:Real,N} ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, mask, Tuple(ones(N)), size(array) )
end

function integralArraySQ!( int_array::Array{T,2}, array::Array{<:Real,2}, mask::Array{<:Real,2}, TL=(1,1), inp_size=size(array) ) where {T<:AbstractFloat}
    TL   = TL .- 1; 
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        int_array[r,c] = T(array[TL[1]+r-1,TL[2]+c-1]*mask[TL[1]+r-1,TL[2]+c-1])^2 + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end
end

function integralArraySQ!( int_array::Array{T,3}, array::Array{<:Real,3}, mask::Array{<:Real,3}, TLF=(1,1,1), inp_size=size(array) ) where {T<:AbstractFloat}
    TLF     = TLF .- 1; 
    h, w, d = inp_size;
    @inbounds for z in 1+1:d+1
        for c in 1+1:w+1
            tmp = 0.0; 
            for r in 1+1:h+1      
                val2          = T(array[TLF[1]r-1,TLF[1]c-1,TLF[1]z-1]*mask[TLF[1]r-1,TLF[1]c-1,TLF[1]z-1])^2
                intArr[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
                tmp          += val2; 
            end 
        end
    end
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TL (TOP-LEFT) AND BR (BOTTOM-RIGHT) CORNERS

function integralArea( intArr::Array{<:AbstractFloat,2}, TL, BR )
	TL   = TL .+ 1 .- 1;
	BR   = BR .+ 1;
	area = intArr[BR[1],BR[2]] - intArr[BR[1],TL[2]] - intArr[TL[1],BR[2]] + intArr[TL[1],TL[2]]
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TLF (TOP-LEFT-FRONT) AND BRB (BOTTOM-RIGHT-BACK) CORNERS

function integralArea( intArr::Array{<:AbstractFloat,3}, TLF, BRB )
	TLF = TLF .+ 1 .- 1; 
	BRB = BRB .+ 1; 
	area  = intArr[BRB[1],BRB[2],BRB[3]] - intArr[TLF[1],TLF[2],TLF[3]]
	area -= intArr[TLF[1],BRB[2],BRB[3]] + intArr[BRB[1],TLF[2],BRB[3]] + intArr[BRB[1],BRB[2],TLF[3]]
	area += intArr[BRB[1],TLF[2],TLF[3]] + intArr[TLF[1],BRB[2],TLF[3]] + intArr[TLF[1],TLF[2],BRB[3]]
    return area
end


# MODIFIED FUNCTIONS TO COPY DATA INTO PADDED ARRAYS FOR MASKED_NSQECC

function copy_inter_masked!( pad_inter, inp1, mask, TLF, inter_size )

    BRB = TLF .+ inter_size .- 1 
    @inbounds pad_inter[ Base.OneTo.( inter_size ) ]  .= inp1[ UnitRange.( TLF, BRB ) ]
    @inbounds pad_inter[ Base.OneTo.( inter_size ) ] .*= mask[ UnitRange.( TLF, BRB ) ]
end

function copy_search_squared!( pad_search, inp2, TLF, inter_size, search_margin )

    search_size = 2 .* inter_size .+ 2 .* search_margin;

    BRB = TLF .+ inter_size .- 1 ; 
 
    outofbounds_TLF  = abs.( min.( TLF .- search_margin .- 1, 0 ) ); 
    search_coords    = UnitRange.( max.( 1, TLF .- search_margin ), min.( size(inp2), BRB .+ search_margin ) );
    padsearch_coords = UnitRange.( outofbounds_TLF .+ 1, outofbounds_TLF .+ length.( search_coords ) );

    @inbounds pad_search[ Base.OneTo.( search_size ) ] .= 0.0
    @inbounds pad_search[ padsearch_coords ]  .= inp2[ search_coords ]; 
    @inbounds pad_search[ padsearch_coords ] .*= inp2[ search_coords ]; 
end

# COMPUTING BBOX FOR INTEGRAL SUMS

function nsqecc_integral_coords( dim_index, size_F, size_G )
    y0 = 1 - min( 0, size_F - dim_index  ); 
    y1 = min( size_G, dim_index );
    return y0, y1
end