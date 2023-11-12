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

# SQUARE THE INPUT WHILE IN-PLACE CONSTRUCTION OF THE INTEGRAL ARRAY

# implementation without a mask for NSQECC

function integralArraySQ!( int_array::AbstractArray{T,N}, 
                           array::AbstractArray{T,N}
                         ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, Tuple( ones(N) ), size( array ) )
end

function integralArraySQ!( int_array::AbstractArray{T,2}, 
                           array::AbstractArray{T,2}, 
                           TL=(1,1), 
                           inp_size=size(array) 
                         ) where {T<:AbstractFloat}
    h, w = inp_size
    @inbounds for c in 1+1:w+1, r in 1+1:h+1
        val2 = array[ ( TL .+ (r,c) .- 2 )... ]^2
        int_array[r,c] = val2 + int_array[r-1,c] + int_array[r,c-1] - int_array[r-1,c-1]
    end

    return nothing
end

function integralArraySQ!( int_array::AbstractArray{T,3},
                           array::AbstractArray{T,3}, 
                           TLF=(1,1,1)::Dims{3}, 
                           inp_size=size(array) 
                         ) where {T<:AbstractFloat}
    h, w, d = inp_size;
    @inbounds for z in 1+1:d+1, c in 1+1:w+1
        tmp = 0.0; 
        for r in 1+1:h+1      
            val2 = array[ ( TLF .+ (r,c,z) .- 2 )... ]^2
            int_array[r,c,z] = val2 + int_array[r,c-1,z] + int_array[r,c,z-1] - int_array[r,c-1,z-1] + tmp;
            tmp += val2; 
        end 
    end

    return nothing
end

# this one accepts a mask for masked_NSQECC

function integralArraySQ!( int_array::Array{T,N}, 
                           array::Array{<:Real,N},
                           mask::Array{<:Real,N}
                         ) where {T<:AbstractFloat,N}
    integralArraySQ!( int_array, array, mask, Tuple(ones(N)), size(array) )
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
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TL (TOP-LEFT) AND BR (BOTTOM-RIGHT) CORNERS

function integralArea( int_arr::Array{<:AbstractFloat,2}, TL, BR )
	TL   = TL .+ 1 .- 1;
	BR   = BR .+ 1;
	area = int_arr[BR[1],BR[2]] - int_arr[BR[1],TL[2]] - int_arr[TL[1],BR[2]] + int_arr[TL[1],TL[2]]
end

# RECOVERING THE SUM OF VALUES WITHIN THE BBOX DEFINED BY TLF (TOP-LEFT-FRONT) AND BRB (BOTTOM-RIGHT-BACK) CORNERS

function integralArea( int_arr::Array{<:AbstractFloat,3}, TLF, BRB )
	TLF = TLF .+ 1 .- 1; 
	BRB = BRB .+ 1; 
	area  = int_arr[BRB[1],BRB[2],BRB[3]] - int_arr[TLF[1],TLF[2],TLF[3]]
	area -= int_arr[TLF[1],BRB[2],BRB[3]] + int_arr[BRB[1],TLF[2],BRB[3]] + int_arr[BRB[1],BRB[2],TLF[3]]
	area += int_arr[BRB[1],TLF[2],TLF[3]] + int_arr[TLF[1],BRB[2],TLF[3]] + int_arr[TLF[1],TLF[2],BRB[3]]
    return area
end

function add_numerator_nsqecc( pad_arr::AbstractArray{T1,2}, 
                               int_arr::AbstractArray{T1,2}, 
                               inter_size,
                               search_margin ) where {T1}

    # fully-overlapping translations
    pad_view = view( pad_arr, UnitRange.( 1, 2 .* search_margin )... ); 

    # 
    ih, iw = inter_size;
    mh, mw = search_margin;
    T, L, D, R = 1:2*mh, 1:2*mw, 1+ih:2*mh+ih, 1+iw:2*mw+iw;

    pad_view .+= view( int_arr, D, R );
    pad_view .+= view( int_arr, T, L );
    pad_view .-= view( int_arr, T, R );
    pad_view .-= view( int_arr, D, L );
end

function add_numerator_nsqecc( pad_arr::AbstractArray{T1,3}, 
                               int_arr::AbstractArray{T1,3}, 
                               inter_size,
                               search_margin ) where {T1}

    # fully-overlapping translations
    pad_view = view( pad_arr, UnitRange.( 1, 2 .* search_margin )... ); 

    # 
    ih, iw, id = inter_size;
    mh, mw, md = search_margin;
    T, L, F, D, R, B = 1:2*mh, 1:2*mw, 1:2*md, 1+ih:2*mh+ih, 1+iw:2*mw+iw, 1+id:2*md+id;

    pad_view .+= view( int_arr, D, R, B );
    pad_view .-= view( int_arr, T, L, F );
    pad_view .-= view( int_arr, T, R, B );
    pad_view .-= view( int_arr, D, L, B );
    pad_view .-= view( int_arr, D, R, F );
    pad_view .+= view( int_arr, D, L, F );
    pad_view .+= view( int_arr, T, R, F );
    pad_view .+= view( int_arr, T, L, B );
end

function fftcc2nsqecc!( pad_F, pad_G, int2_G, sumF2, size_F, size_G  )

    # 1-. identifying the range of fully-overlapping translations
    marg_G = div.( size_G .- size_F, 2 ); 
    fovp = UnitRange.( 1, 2 .* marg_G  ); 

    # 2-. Manipulating pad_F, which initially contains sum(G*F) for all translations
    pad_F[ fovp... ] .*= -2;                              # - 2sum(G*F)
    pad_F[ fovp... ] .+= sumF2;                           # sum(F²) - 2sum(G*F)
    add_numerator_nsqecc( pad_F, int2_G, size_F, marg_G ) # sum(F²) + sum(G²) - 2sum(G*F)
                                            
    # 3.2-. Computing denominator within pad_G. 
    pad_G[ fovp... ] .= 0.0
    add_numerator_nsqecc( pad_G, int2_G, size_F, marg_G ) # sum(G²)
    pad_G[ fovp... ] .= sqrt.( pad_G[ fovp... ] )         # √sum(G²)
    pad_G[ fovp... ] .*= sqrt( sumF2 )                    # √sum(F²)√sum(G²)

    # 3.3-. numerator / denominator within pad_F
    pad_F[ fovp... ] ./= pad_G[ fovp... ];

    return nothing
end

# MODIFIED FUNCTIONS TO COPY DATA INTO PADDED ARRAYS FOR MASKED_NSQECC

function copy_inter_masked!( pad_inter, inp1, mask, TLF, isize )

    BRB = TLF .+ isize .- 1 
    pad_inter .= 0.0
    @inbounds pad_inter[ Base.OneTo.( isize )... ]  .= inp1[ UnitRange.( TLF, BRB )... ]
    @inbounds pad_inter[ Base.OneTo.( isize )... ] .*= mask[ UnitRange.( TLF, BRB )... ]
end

function copy_inter_squared!( pad_inter, inp1, TLF, isize )

    BRB = TLF .+ isize .- 1 
    pad_inter .= 0.0
    @inbounds pad_inter[ Base.OneTo.( isize )... ]  .= inp1[ UnitRange.( TLF, BRB )... ]
    @inbounds pad_inter[ Base.OneTo.( isize )... ] .*= inp1[ UnitRange.( TLF, BRB )... ]
end

function copy_search_squared!( pad_search, inp2, TLF, isize, smarg )

    BRB = TLF .+ isize .- 1 ; 
 
    outofbounds_TLF  = abs.( min.( TLF .- smarg .- 1, 0 ) ); 
    search_coords    = UnitRange.( max.( 1, TLF .- smarg ), min.( size(inp2), BRB .+ smarg ) );
    padsearch_coords = UnitRange.( outofbounds_TLF .+ 1, outofbounds_TLF .+ length.( search_coords ) );

    pad_search .= 0.0
    @inbounds pad_search[ padsearch_coords... ]  .= inp2[ search_coords... ]; 
    @inbounds pad_search[ padsearch_coords... ] .*= inp2[ search_coords... ]; 
end