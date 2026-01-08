using Statistics 

begin ########################################################### UTILS 

    begin #####################

        """ vfsize( VF ) """
        vfsize( inp::AbstractArray ) = Tuple( size( inp )[2:end] )

        """ vfsize( (U,V,W) ) """
        # NOTE: This might be good to add '@assert all( size.(inp) .== (size(inp[1]),) )'
        vfsize( inp::AbstractArray... ) = size( inp[1] )

        """ num_components( VF ) """
        num_components( inp::AbstractArray ) = size( inp, 1 )

        """ num_components( (U, V, W) ) """
        num_components( inp::AbstractArray... ) = length( N )

        """ apply_each_dimension( op, ROI, VF ) """
        function apply_each_dimension( op, ROI, inp::AbstractArray )
            n_dims = num_components( inp ); 
            output = zeros( eltype(inp), n_dims )
            for c in 1:n_dims
                output[c] = op( inp[ c, ROI... ] )
            end
            return output
        end

        """ apply_each_dimension( op, ROI, (U,V,W) ) """
        function apply_each_dimension( op, ROI, inp::AbstractArray... )
            n_dims = num_components( inp ); 
            output = zeros( eltype(inp[1]), n_dims )
            for c in 1:n_dims
                output[c] = op( inp[c][ ROI... ] )
            end
            return output
        end
    end

    """
        Most post-processing functions require computing some quantity for all vector field components 
        within a certain square region around each position of the vector field. This function can be
        used for exactly that. It is flexible enough to deal with vector fields provided as individual 
        arrays ( U, V or U, V, W ), as well as vector fields combined into one array, which is the  
        format that quickPIV.PIV( ... ) returns.

        Example of computing the average vector withim a radius of (4,4) around coordinate (10,5): 

            avg_vec_2D = localOp( (10,5), (4,4), U, V, op=(x)->(Statistics.mean(x)) )
            avg_vec_2D = localOp( (10,5), (4,4), VF, op=(x)->(Statistics.mean(x)) )

            avg_vec_3D = localOp( (10,5,10), (4,4,4), U, V, W, op=(x)->(Statistics.mean(x)) )
            avg_vec_3D = localOp( (10,5,10), (4,4,4), VF, op=(x)->(Statistics.mean(x)) )
    """
    function localOp( 
        coords::NTuple{N,Int}, 
        radius::dimable, 
        vectorfield::AbstractArray...; 
        op=(x)->(Statistics.mean(x)) 
    ) where {N}

        vf_dims   = num_components( vectorfield )
        vf_size   = vfsize( vectorfield )
        radii     = toDims3( radius )[1:N]
        mincoords = max.(    1   , coords .- radii );
        maxcoords = min.( vf_size, coords .+ radii );
        ROI       = UnitRange.( mincoords, maxcoords ); 
        result    = apply_each_dimension( op, ROI, vectorfield )
        return result
    end

    function localOp( coords::NTuple{N,Int}, radius::dimable, arrays::Array{T,N}...; op=(x)->(Statistics.mean(x)) ) where {T,N}
        radii     = toDims3( radius )[1:N]
        mincoords = max.( 1, coords .- radii );
        maxcoords = min.( size(arrays[1]), coords .+ radii );
        result    = zeros( T, length( arrays ) );
        for c in 1:length( arrays )
            result[ c ] = op( arrays[c][ UnitRange.( mincoords, maxcoords )... ] );
        end
        return result
    end

    function localOp( coords::NTuple{N1,Int}, radius::dimable, VF::Array{T,N2}; op=(x)->(Statistics.mean(x)) ) where {T,N1,N2}
        radii     = toDims3( radius )[1:N1]
        mincoords = max.( 1, coords .- radii );
        maxcoords = min.( size(VF)[2:end], coords .+ radii );
        result    = zeros( T, size(VF,1) );
        for c in 1:size(VF,1)
            result[ c ] = op( VF[ c, UnitRange.( mincoords, maxcoords )... ] );
        end
        return result
    end

    function localOp( coords::NTuple{N,Int}, radius::dimable, data::Array{T,N}; op=(x)->(Statistics.mean(x)) ) where {T,N}
        radii     = toDims3( radius )[1:N]
        mincoords = max.( 1, coords .- radii );
        maxcoords = min.( size(data), coords .+ radii );
        result    = op( data[ UnitRange.( mincoords, maxcoords )... ] );
        return result
    end

    function apply_mask!( mask::Array{T1,N}, arrays::Array{T2,N}... ) where {T1,T2,N}
        for c in 1:length(arrays)
            arrays[c] .*= mask
        end
        return nothing
    end

    function apply_mask!( mask::Array{T1,N1}, VF::Array{T2,N2} ) where {T1,T2,N1,N2}
        for c in 1:size(VF,1)
            VF[c, UnitRange.(1,size(mask))...] .*= mask
        end
        return nothing
    end
end

begin ########################################################### COMPUTING MAGNITUDES

    """
        Computes the "velocity", "speed" or "magnitude" of each vector, where 
        each component is stored in a separate array. You can use any of the 
        three terms, whichever you prefer: 
        
            M = magnitudes( U, V ); # 2D
            M = magnitudes( U, V, W ); # 3D

            M = speeds( U, V ); # 2D
            M = speeds( U, V, W ); # 3D

            M = velocities( U, V ); # 2D
            M = velocities( U, V, W ); # 3D
    """

    speeds( arrays::Array{T,N}...; typ=T ) where {T,N} = magnitudes( arrays..., typ=typ )

    velocities( arrays::Array{T,N}...; typ=T ) where {T,N} = magnitudes( arrays..., typ=typ )

    function magnitudes( arrays::Array{T,N}...; typ=T ) where {T,N}

        magnitudes = zeros( typ, size( arrays[1] ) )
        magnitudes!( magnitudes, arrays... )
        return magnitudes
    end

    function magnitudes!( out::Array{T1,N}, arrays::Array{T2,N}... ) where {T1,T2,N}

        @assert size(out) == size(arrays[1])
        numcomponents = length( arrays );
        @assert 1 < numcomponents < 4

        @inbounds for e in 1:length( arrays[1] )
            sum2 = 0.0;
            @simd for c in 1:numcomponents
                sum2 += arrays[c][e]^2
            end
            out[e] = convert( T1, sqrt( sum2 ) )
        end

        return nothing
    end

    """
        Computes the "velocity", "speed" or "magnitude" of each vector, where 
        all component is combined in a single array. This is how quickPIV returns
        the PIV vector fields. Again, you can use any of the three terms, whichever
        you prefer: 
        
            M = magnitudes( VF ); 
            M = speeds( VF ); 
            M = velocities( VF );
    """

    speeds( VF::Array{T,N}; typ=T ) where {T,N} = magnitudes( VF, typ=typ ); 

    velocities( VF::Array{T,N}; typ=T ) where {T,N} = magnitudes( VF, typ=typ ); 

    function magnitudes( VF::Array{T,N}; typ=T ) where {T,N}
        @assert 1 < size( VF,1 ) < 4
        magnitudes = zeros( typ, size(VF)[2:end]... )
        magnitudes!( magnitudes, VF )
        return magnitudes
    end

    function magnitudes!( magnitudes::Array{T1,N1}, VF::Array{T2,N2} ) where {T1,T2,N1,N2}

        @assert N2 == N1 + 1 
        @assert 1 < size( VF,1 ) < 4

        for index in CartesianIndices( magnitudes )
            magnitudes[ index ] = sqrt( sum( VF[ :, index ] .^ 2 ) )
        end
        return nothing
    end

    # TODO: implement for U, V, W...
    """
        It is usually desired to remove vectors which are abnormally large compared to the vectors
        around them. In other words, we need to check the average/minimum magnitude around each
        vector and to remove vector's whose magnitude is much greater. 
    """
    function local_magnitude_mask( ratio, VF::Array{T,N}; radius=4 ) where {T,N}

        mags = magnitudes( VF ); 
        mask = zeros( Bool, size(mags) ); 
        
        for ci in CartesianIndices( mags )
            sum_mag  = localOp( Tuple( ci ), radius, mags, op=(x)->(sum(x)) ) - mags[ci]
            sum_N    = localOp( Tuple( ci ), radius, mags, op=(x)->(length(x)-1) )
            mean_mag = sum_mag/sum_N
            mask[ ci ] = mags[ci]/mean_mag < ratio
        end
        VF_c = copy( VF ); 
        for c in 1:size(VF,1)
            VF_c[c,:,:] .*= mask
        end
        return VF_c, Bool.( mask .== 0 )
    end

end

begin ########################################################### COMPUTING COLLECTIVENESS

    """
        Computes the local similarity for each vector in a vector field, where each component
        of the vector field is given in its own array.
            
        In other words: for each vector it looks at its neighboring vectors and counts the
        proportion of the neighbouring vectors that have a similar direction. In particular, 
        "having a similar direction" means that the angle between the vectors is smaller than 
        "max_angle". 

        The first parameter is the size of the neighboring region. 

        Usage: 

            coll = collectiveness( 4, U, V ); # 2D similarity in a 9x9 region around each vector
            coll = collectiveness( (3,3,2), U, V, W ); # 3D simliarity in a 7x7x5 region around each vector
    """
    function collectiveness( radius::dimable, arrays::Array{T,N}...; max_angle=40 )  where {T,N}
        out = zeros( Float32, size(arrays[1]) ); 
        collectiveness!( out, radius, arrays..., max_angle=max_angle ); 
        return out
    end

    function collectiveness!( out, radius::dimable, arrays::Array{T,N}...; max_angle=40 ) where {T,N}

        @assert size( out ) == size( arrays[1] ); 
        ncomponents = length( arrays ); 
        @assert 1 < ncomponents < 4

        VF_size = size( arrays[1] ); 
        min_dot = cosd( max_angle ); 
        radii   = toDims3( radius )[1:N]

        # we will need the magnitudes to normalize each vector
        M = magnitudes( arrays ); 

        for index in CartesianIndices( arrays[1] )

            # normalized vector at "index"
            mag  = M[ index ]
            nvec = [ arrays[c,index]/mag for c in 1:ncomponents ]

            # counting the number of similar neighbours around "index"
            local_region = UnitRange.( max.( 1, Tuple(index) .- radii ),  min.( VF_size, Tuple(index) .+ radii ) )
            N_neighbors  = prod( length.( local_region ) ); 
            N_similar    = 0 
            for local_index in CartesianIndices( local_region )
                mag_  = M[ local_index ]
                nvec_ = [ arrays[c][local_index]/mag_ for c in 1:ncomponents ]
                dot_  = sum( vec .* nvec_ )
                N_similar += dot_ > min_dot
            end

            out[index] = N_similar / N_neighbors
            # TODO: integral dot product as a similarity measure
        end
        return nothing
    end

    """
        Computes the local similarity for each vector in a vector field, where all components
        of the vector field are combined into one array (VF).

        Usage: 

            coll = collectiveness( 4, U, V ); # 2D similarity in a 9x9 region around each vector
            coll = collectiveness( (3,3,2), U, V, W ); # 3D simliarity in a 7x7x5 region around each vector
    """
    function collectiveness( radius::dimable, VF::Array{T,N}; max_angle=40 )  where {T,N}
        out = zeros( Float32, size(VF)[2:end] ); 
        collectiveness!( out, radius, VF, max_angle=max_angle ); 
        return out
    end

    function collectiveness!( out::Array{T1,N1}, radius::dimable, VF::Array{T2,N2}; max_angle=40 ) where {T1,T2,N1,N2}

        @assert 1 < size(VF,1) < 4
        @assert N2 == N1 + 1
        @assert all( [ size( out, i ) == size( VF, 1+i ) for i in 1:N1 ] ); 


        VF_size = size( VF )[2:end]; 
        min_dot = cosd( max_angle ); 
        radii   = toDims3( radius )[1:N1]

        # we will need the magnitudes to normalize each vector
        M = magnitudes( VF ); 

        for index in CartesianIndices( M )

            # normalized vector at "index"
            mag  = M[ index ]
            nvec = VF[ :, index ] ./ mag

            # counting the number of similar neighbours around "index"
            local_region = UnitRange.( max.( 1, Tuple(index) .- radii ),  min.( VF_size, Tuple(index) .+ radii ) )
            N_neighbors  = prod( length.( local_region ) ); 
            N_similar    = 0 
            for local_index in CartesianIndices( local_region )
                mag_  = M[ local_index ]
                nvec_ = VF[ :, local_index ] ./ mag_
                dot_  = sum( nvec .* nvec_ )
                N_similar += dot_ > min_dot
            end

            out[index] = N_similar / N_neighbors
        end
        return nothing
    end
end

begin ########################################################### AVERAGING

    """
        spatial smoothing of vector fields stored as individual arrays (U,V or U,V,W)
    """
    function average( radius::dimable, arrays::Array{T,N}...) where {T,N}

        ncomponents = length( arrays ); 
        @assert 1 < ncomponents < 4

        avg_arrays = [ copy( array ) for array in arrays ]

        VF_size = size( arrays[1] ); 
        radii   = toDims3( radius )[1:N]
    
        for index in CartesianIndices( VF_size )

            avg_vector = localOp( Tuple(index), radius, arrays..., op=(x)->(Statistics.mean(x)) )
            for c in 1:ncomponents
                avg_arrays[c][index] = avg_vector[c]
            end
        end

        return avg_arrays
    end

    """
        spatial smoothing of vector fields where all components are combined in a single
        array.
    """
    function average( radius::dimable, VF::Array{T,N} ) where {T,N}

        ncomponents = size( VF, 1 ); 
        @assert 1 < ncomponents < 4

        avg_VF = copy( VF )

        VF_size = size( VF )[2:end]; 
        radii   = toDims3( radius )[1:N-1]
    
        for index in CartesianIndices( VF_size )

            avg_vector = localOp( Tuple(index), radius, VF, op=(x)->(Statistics.mean(x)) )
            avg_VF[:,index] .= avg_vector
        end

        return avg_VF
    end

    """
        TODO: space time averaging
    """
    function average_NDT( radius1::dimable, radius2::Int, arrays::Array{T,N}... )  where {T,N}
    
        avg_arrays = [ copy( array ) for array in arrays ]

        num_timepoints = size(arrays[1])[end]
        spatial_dims = size(arrays[1])[1:end-1]

        for index in CartesianIndices( arrays[1] )
            avg_vec1 = localOp( index[1:end-1], (radius1,0), arrays..., op=(x)->(Statistics.mean(x)))
            avg_vec2 = localOp( index[1:end-1], (0,0,0,radius2), arrays..., op=(x)->(Statistics.mean(x)))
        end

        return avg_arrays
    end

    function average_NDT( radius1::dimable, radius2::Int, VF::Array{T,N} )  where {T,N}
    
        avg_arrays = copy( VF )

        num_timepoints = size(VF)[end]
        spatial_dims = size(VF)[2:end-1]

        for index in CartesianIndices( arrays[1] )
            avg_vec1 = localOp( index[1:end-1], (radius1,0), VF, op=(x)->(Statistics.mean(x)))
            avg_vec2 = localOp( index[1:end-1], (0,0,0,radius2), VF, op=(x)->(Statistics.mean(x)))
        end

        return avg_arrays
    end

end

begin ########################################################### COLLECTIVENESS-GUIDED AVERAGING

    """ Spatial averaging + similarity thresholding combo 
    """
    function similarityAveraging( radius::dimable, arrays::Array{T,N}...; 
                                  max_angle=20, 
                                  normalize=true,
                                  scale_by_similarity=true,
                                  similarity_exponent=2  ) where {T,N}
    
        ncomponents = length(arrays)
        @assert 1 < ncomponents < 4
    
        radii   = toDims3( radius )[1:N]
        min_dot = cosd( max_angle )
        VF_size = size( arrays[1] ); 
        avg_arrays = [ copy(array) for array in arrays ]
    
        M = magnitudes( arrays... )
    
        for index in CartesianIndices( arrays[1] )
    
            mag = M[ index ]
            if mag == 0
                continue
            end
    
            # normalized vector at "index"
            nvec = [ arrays[c][index]/mag for c in 1:ncomponents ]
    
            # counting the number of similar neighbours around "index"
            local_region = UnitRange.( max.( 1, Tuple(index) .- radii ),  min.( VF_size, Tuple(index) .+ radii ) )
            N_neighbors  = prod( length.( local_region ) ); 
            N_similar    = 0 
            avg_vector   = [ 0.0 for c in 1:ncomponents ]; 
            for local_index in CartesianIndices( local_region )
                mag_  = M[ local_index ]
                nvec_ = [ arrays[c][local_index]/mag_ for c in 1:ncomponents ]
                dot_  = sum( nvec .* nvec_ )
    
                avg_vector .+= nvec_ .* mag_ .* ( dot_ > min_dot )
                N_similar   += dot_ > min_dot
            end
    
            fac = 1
            fac /= ( normalize ) ? sqrt( sum( avg_vector .^ 2 ) ) :  prod( length.( local_region ) ); 
            fac *= ( scale_by_similarity ) ? (N_similar/N_neighbors)^similarity_exponent : 1; 
    
            for c in 1:ncomponents
                avg_arrays[c][index] = avg_vector[c] * fac
            end
        end
    
        return avg_arrays
    end

    """ Spatial averaging + similarity thresholding combo """
    function similarityAveraging( radius::dimable, VF::Array{T,N}; 
                                  max_angle=20, 
                                  normalize=true,
                                  scale_by_similarity=true,
                                  similarity_exponent=2  ) where {T,N}
    
        ncomponents = size(VF,1)
        @assert 1 < ncomponents < 4
    
        radii   = toDims3( radius )[1:N-1]
        min_dot = cosd( max_angle )
        VF_size = size( VF )[2:end]
        avg_VF  = copy( VF )
    
        M = magnitudes( VF )
    
        for index in CartesianIndices( VF_size )
    
            mag = M[ index ]
            if mag == 0
                continue
            end
    
            # normalized vector at "index"
            nvec = VF[:,index]./mag
    
            # counting the number of similar neighbours around "index"
            local_region = UnitRange.( max.( 1, Tuple(index) .- radii ),  min.( VF_size, Tuple(index) .+ radii ) )
            N_neighbors  = prod( length.( local_region ) ); 
            N_similar    = 0 
            avg_vector   = [ 0.0 for c in 1:ncomponents ]; 
            for local_index in CartesianIndices( local_region )
                mag_  = M[ local_index ]
                nvec_ = VF[:,local_index]./mag_
                dot_  = sum( nvec .* nvec_ )
    
                avg_vector .+= nvec_ .* mag_ .* ( dot_ > min_dot )
                N_similar   += dot_ > min_dot
            end
    
            fac  = 1
            fac /= ( normalize ) ? sqrt( sum( avg_vector .^ 2 ) ) :  1; 
            fac *= ( scale_by_similarity ) ? (N_similar/N_neighbors)^similarity_exponent : 1; 
    
            avg_VF[:,index] .= avg_vector .* fac
        end
    
        return avg_VF
    end

end

begin ########################################################### FILTERING

    begin ###### REPLACEMENT SCHEMES 
        """
            The replacement functions take a coordinate, a radius and the PIV vector field components,
            and return the mean or median vector around the input coordinate. zeroReplace simply returns
            a zero-vector.

            The radius can be a single number, in which case all dimensions will use the same radius. 
            Alternatively, one can provide a different radius for each dimension.
        """

        function zeroReplace( coords::NTuple{N,Int}, radius::dimable, arrays::Array{T,N}... ) where {T,N}
            return zeros( T, length(arrays) );
        end

        function zeroReplace( coords::NTuple{N,Int}, radius::dimable, VF::Array{T,M} ) where {T,N,M}
            return zeros( T, size(VF,1) );
        end

        function medianReplace( coords::NTuple{N,Int}, radius::dimable, arrays::Array{T,N}... ) where {T,N}
            return localOp( coords, radius, arrays..., op=(x)->(Statistics.median(x)) )
        end

        function medianReplace( coords::NTuple{N,Int}, radius::dimable, VF::Array{T,M} ) where {T,N,M}
            return localOp( coords, radius, VF, op=(x)->(Statistics.median(x)) )
        end

        function meanReplace( coords::NTuple{N,Int}, radius::dimable, arrays::Array{T,N}... ) where {T,N}
            return localOp( coords, radius, arrays..., op=(x)->(Statistics.median(x)) )
        end

        function meanReplace( coords::NTuple{N,Int}, radius::dimable, VF::Array{T,M}... ) where {T,N,M}
            return localOp( coords, radius, VF, op=(x)->(Statistics.median(x)) )
        end

        function replace!( coords, replacement, filtered::Array{T,N}... ) where {T,N}
            for c in 1:length(filtered)
                filtered[c][coords] = replacement[c]
            end
            return nothing
        end

        function replace!( coords, replacement, filtered::Array{T,N} ) where {T,N}
            filtered[:,coords] .= replacement
            return nothing
        end
    end

    begin ###### FILTERING TYPES: THRESHOLD OR STATISTICAL ( standard deviations from the mean )

        """
            Creates a mask of elements whose value is > n standard deviations away from the mean. 
            The exact quantity that we compute is dynamically chosen by "fun". By default, "fun"
            computes the magnitude of the input vector field.
            
            mask = STD_masking( 1.8, U, V [, W ] ) 
            mask = STD_masking( 2.2, VF, op=(VF)->(quickPIV.similarity(VF)) )
        """
        function STD_masking( n, arrays::Array{T,N}...; 
                              fun=(arrays)->(magnitudes(arrays...)) ) where {T,N}

            map  = fun( arrays... ) 
            return STD_masking( n, map, fun=(x)->(x) )
        end

        #=
          This function is meant to accept a single array that contains all vector field components.
          However, we can use it with arbitrary arrays, such as the signal-to-noise ratio or vector 
          magnitudes, as long as we set "fun" to return the inputs themselves: (x)->(x) 
        =#
        function STD_masking( n, VF::Array{T,N}; 
                              fun=(VF)->(magnitudes(VF)) ) where {T,N}

            map  = fun( VF ) 
            mask = zeros( Bool, size(map) )
            mean = Statistics.mean( map );
            std  = Statistics.std( map, mean=mean ) ;
            @simd for e in 1:length( map )
                @inbounds mask[ e ] = abs( map[e] - mean ) > n*std
            end
            return mask
        end

        """
            Applies "STD_masking" and replaces the resulting mask .== 1 with the
            chosen replacement scheme.
        """
        function STD_filtering( n, arrays::Array{T,N}...; 
                                fun=(x)->(magnitudes(x...)), 
                                replaceFun=meanReplace, 
                                radius=2  ) where {T,N}

            mask   = STD_masking( n, arrays..., fun=fun ); 
            coords = CartesianIndices( mask )[ mask .== 1 ]; 
            filtered = [ copy(array) for array in arrays ]
            for index in coords
                replacement = replacefun( Tuple(index), radius, arrays... )
                replace!( index, replacement, filtered )
            end
            return filtered
        end

        function STD_filtering( n, VF::Array{T,N}; 
                                fun=(x)->(magnitudes(x)), 
                                replaceFun=meanReplace, 
                                radius=2  ) where {T,N}

            mask   = STD_masking( n, VF, fun=fun ); 
            coords = CartesianIndices( mask )[ mask .== 1 ]; 
            filtered = copy( VF )
            for index in coords
                replacement = replaceFun( Tuple(index), radius, VF )
                replace!( index, replacement, filtered )
            end
            return filtered
        end

        """
            Creates a mask of elements whose value is > or < threshold. 

            The exact quantity that we compute is dynamically chosen by "fun". By default, "fun"
            computes the magnitude of the input vector field.
        """

        function TH_masking( th, arrays::Array{T,N}...; 
                             fun=(arrays)->(magnitudes(arrays...)), 
                             cmp=(x,th)->(x>th) ) where {T,N}

            map  = fun( arrays... )
            return TH_masking( th, map, fun=(x)->(x) )
        end

        function TH_masking( th, VF::Array{T,N};
                             fun=(array)->(magnitudes(array)), 
                             cmp=(x,th)->(x>th) ) where {T,N}

            map = fun( VF )
            mask = zeros( Bool, size(map) )
            @simd for e in 1:length( map )
                @inbounds mask[ e ] = cmp( map[e], th )
            end
            return mask
        end

        """
            Computes a mask through tresholding and replaces the values where mask .== 1
            with the desired replacement scheme.
        """
        function TH_filtering( th, arrays::Array{T,N}...; 
                               fun=(x)->(magnitudes(x...)), 
                               cmp=(x,th)->(x>th), 
                               replaceFun=meanReplace, 
                               radius=2  ) where {T,N}

            mask   = TH_masking( th, arrays..., fun=fun, cmp=cmp ); 
            coords = CartesianIndices( mask )[ mask .== 1 ]; 
            filtered = [ copy(array) for array in arrays ]
            for index in coords
                replacement = replacefun( Tuple(index), radius, arrays... )
                replace!( index, replacement, filtered )
            end
            return filtered
        end

        function TH_filtering( th, VF::Array{T,N}; 
                               fun=(x)->(magnitudes(x)), 
                               cmp=(x,th)->(x>th), 
                               replaceFun=meanReplace, 
                               radius=2  ) where {T,N}

            mask   = TH_masking( th, VF, fun=fun, cmp=cmp ); 
            coords = CartesianIndices( mask )[ mask .== 1 ]; 
            filtered = copy( VF )
            for index in coords
                replacement = replacefun( Tuple(index), radius, VF )
                replace!( index, replacement, filtered )
            end
            return filtered
        end
    end

    """
        Remove vectors with low signal-to-noise.
    """
    function remove_low_SN( SN, threshold, arrays::Array{T,N}...;
                            replace::Function=zeroReplace, radius=1 ) where {T,N}

        filtered = TH_filtering( threshold, arrays...; 
                                 fun=(x)->(SN), 
                                 cmp=(x,th)->(x<th),
                                 replace=replace, 
                                 radius=radius )
        return filtered
    end

    function remove_low_SN( SN, threshold, VF::Array{T,N};
                            replace::Function=zeroReplace, radius=1 ) where {T,N}

        filtered = TH_filtering( threshold, VF; 
                                 fun=(x)->(SN), 
                                 cmp=(x,th)->(x<th),
                                 replace=replace, 
                                 radius=radius )
        return filtered
    end

    """
        TODO: magnitude filtering
    """
    function filter_by_magnitudes( th_or_n,  )

    end


end

begin ########################################################### DIVERGENCE

    """
        We compute divergence by cross-correlating a vector field kernel containing a 
        "sink" pattern with the normalized PIV vector field. This results in high
        positive values in regions of the PIV vector fields that resemble the "sink"
        pattern and high negative values in regions that resemble a "source". 

        In any case, the "sink" kernel contains normalized vectors pointing towards
        the center of the kernel. Therefore, we need to compute the magnitudes at 
        at position of the kernel, so that we can normalize the components (U,V or
        U,V,W) of the sink.
    """
    function normalize_sink( radii; typ=Float32, scales=ones(eltype(radii),length(radii)), circle=false )

        sink_size = 2 .* radii .+ 1; 
        center    = div.( sink_size .+ 1 , 2 )
        
        vec_magnitudes = zeros( typ, sink_size ); 
        max_radius = maximum( radii ); 
        
        for ci in CartesianIndices( vec_magnitudes )
            vec_magnitudes[ci] = sqrt( sum( ( center .* scales .- Tuple( ci ) .* scales ).^2 ) ); 
            if circle
                vec_magnitudes[ci] *= vec_magnitudes[ci] .<= max_radius
            end
        end
        vec_magnitudes[ center... ] = 1; 
        
        return vec_magnitudes
    end

    """
        Cross-correlating the "sink" kernel with the input vector field can be done
        by cross-correlating each component separately, and adding everything together.
        Thus, we can first cross-correlate the U components of the "sink" and the PIV
        vectorfield... to which we add the cross-correlation of the V components... 
        followed by the W components (in 3D). 

        Therefore, we only need one component of the "sink" kernel at a time. The 
        function below computes the "sink" values at the desired dimension, and
        stores them into the preallocated array "sink", for in-place operation.
    """
    function sink!( sink::AbstractArray{T,N}, vec_magnitudes, dim ) where {T,N}
        
        @assert all( isodd.( size(sink) ) );
        
        center = div.( size(sink) .+ 1 , 2 ); 
        
        for ci in CartesianIndices( sink )
            if vec_magnitudes[ci] == 0
                sink[ci] = 0
            else
                sink[ci] = ( center[dim] - ci[dim] ) / vec_magnitudes[ci]; 
            end
        end

        sink[ isnan.( sink ) ] .= 0.0
        
        return nothing
    end

    """
        Out-of-place computation of the "sink" pattern for the desired dimension.

        sink_u = sink( radii, 1 );
        sink_v = sink( radii, 2 ); 
        sink_w = sink( radii, 3 );
    """
    function sink( radii, dim=1 )
        sinkM = normalize_sink( radii )
        sink  = copy( sinkM )
        sink!( sink, sinkM, dim )
        return sink
    end

    """
        The input vector fild needs to be normalized, so that cross-correlation of
        the "sink" pattern with the vector field corresponds to the sum of normalized
        dot products between the two, which measures the angle-similarity of the
        between the two. 
    """
    function normalize_vectorfield( VF; normalize=true, min_speed=0 )

        VF_copy = copy( VF ); 
        M = multi_quickPIV.magnitudes( VF );

        if min_speed > 0
            for c in 1:size(VF,1)
                VF_copy[ c, UnitRange.(1,size(VF)[2:3])... ] .*= M .> min_speed
            end
        end

        if normalize
            M[ M .== 0 ] .= 1; 
            for c in 1:size(VF,1)
                VF_copy[ c, UnitRange.(1,size(VF)[2:3])... ] ./= M
            end
        end

        return VF_copy    
    end

    """
        Compute divergence by cross-correlating a sink pattern of size "radii" with 
        the input vector field.     

        NOTE: divergence is actually computing convergence at the moment...
    """
    function divergence( radii, VF::Array{T,N}; normalize=false, min_speed=0, divide_by="N", circle=false ) where {T,N}

        # normalizing vector field. 
        VF_norm = normalize_vectorfield( VF, normalize=normalize, min_speed=min_speed ); 
        
        # Preallocating memory for the sink pattern and precomputing the magnitudes at each sink position.
        sink_magnitudes = normalize_sink( radii, typ=T, circle=circle )
        sink = zeros( T, size(sink_magnitudes) )

        mags     = magnitudes( VF_norm ); 
        sum_mags = FFTCC_crop( ones( T, size( sink_magnitudes ) ), mags )
        
        # output
        divergence = zeros( T, size( VF )[2:end] )
        
        # cross-correlation variables
        isize = size( sink ); 
        ssize = size( VF )[2:end]; 
        prec  = sizeof( T ) * 8; 
        tmp_data = multi_quickPIV.allocate_tmp_data( multi_quickPIV.FFTCC(), isize, ssize, precision=prec, unpadded=false, good_pad=true )
        
        # cross-correlating each dimension separately and adding the results into divergence
        n_components = size( VF, 1 )
        for c in 1:n_components
            sink!( sink, sink_magnitudes, c )
            tmp_data[1] .= 0.0
            tmp_data[2] .= 0.0
            tmp_data[1][UnitRange.(1,isize)...] .= sink; 
            tmp_data[2][UnitRange.(1,ssize)...] .= VF_norm[c,UnitRange.(1,ssize)...]
            
            multi_quickPIV._FFTCC!( tmp_data... )
            
            r2c_pad = size(tmp_data[1]) .- tmp_data[end]; 
            Base.circshift!(  view( tmp_data[2], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad )... ),
                              view( tmp_data[1], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad )... ),
                              div.( isize, 2 ) ); 
            divergence .+= tmp_data[2][ UnitRange.( 1, size(divergence) )... ]                
        end

        if divide_by == "N"
            # dividing by the number of vectors... 
            divergence ./= sum( sink_magnitudes .> 0 )
        elseif divide_by == "mags"
            # dividing by sum of vector magnitudes... which is the number of normalized vectors anyways
            divergence ./= sum_mags; 
        end
        
        multi_quickPIV.destroy_fftw_plans( multi_quickPIV.FFTCC(), tmp_data )
        
        return divergence
    end


    """
        The input vector field is a matrix containing the U and V [ and W ]components in its first dimension. 
        U = VF[ 1, :, : ], VF = [ 2, :, : ];
        
        The y derivative is computed on U along the rows: ( U[ row+1, col ] - U[ row-1, col ] )/( 2 ) 
            == ( VF[ 1, row+1, col ] - VF[ 1, row-1, col ] )/( 2 ). 

        The x derivative is computed on V along the columns: ( V[ row+1, col ] - V[ row-1, col ] )/( 2 )
            ==  ( VF[ 2, row, col+1 ] - VF[ 2, row, col-1 ] )/( 2 ). 

        https://de.mathworks.com/help/matlab/ref/divergence.html#mw_b9736035-36f1-4f34-81f7-d530a9216a37
    """
    function matlab_divergence( VF::Array{T,N} ) where {T,N}
        
        Ndims   = size( VF, 1 ); 
        vf_size = size( VF )[2:end]; 

        output  = zeros( Float32, vf_size ); 

        for ci in CartesianIndices( vf_size )

            coord = Tuple( ci ); 

            # adding the partial derivative for each dimension (Ndims)
            for i in 1:Ndims
                offset = zeros( Int, Ndims ); 
                offset[i] = 1; 
                coord_minus = max.( 1, min.( vf_size, coord .- offset ) )
                coord_plus  = max.( 1, min.( vf_size, coord .+ offset ) )
                partial_der = Float32( ( VF[ i, coord_plus... ] .- VF[ i, coord_minus... ] )/abs( coord_minus[i] - coord_plus[i] ) )
                output[ ci ] += partial_der
            end
        end
        
        return output
    end

    # TODO: divergence with vector fields as separate arrays

    """
        
    """
    function multiscale_divergence( radii, VF::Array{T,N}; min_value=0.2, normalize=true, min_speed=0, divide_by="N" ) where {T,N}

        max_divergence_  = zeros( T, size(VF)[2:end] )
        max_convergence_ = zeros( T, size(VF)[2:end] )
        max_divergence_scale  = zeros( Int, size(VF)[2:end] )
        max_convergence_scale = zeros( Int, size(VF)[2:end] )

        for rad in radii

            # Multiplying by -1 because divergence is computing convergence at the moment
            divergence_ = -1 .* divergence( rad, VF, normalize=normalize, min_speed=min_speed, divide_by=divide_by ); 

            for i in 1:length( divergence_ )
                
                divg = divergence_[i]
                conv = -1 * divg; 

                # 
                if ( divg > min_value ) && ( divg > max_divergence_[i] )
                    max_divergence_scale[i] = rad[1]
                    max_divergence_[i] = divg
                end

                #
                if ( conv > min_value ) && ( conv > max_convergence_[i] )
                    max_convergence_scale[i] = rad[1]
                    max_convergence_[i] = conv
                end
            end
        end

        return max_divergence_scale, max_divergence_, max_convergence_scale, max_convergence_
    end

end

begin ########################################################### PSEUDO-TRAJECTORIES 

    """ PIV Trajectories, work started by Michelle Gottlieb  """

    function PIVtrajectories( U::Array{P,4}, V::Array{P,4}, W::Array{P,4}, T0, T1, numpoints; 
                            subregion=( 1:-1, 1:-1, 1:-1 ), scale=(1,1,1) ) where {P<:Real}

        numT = T1 - T0;
        TrajectoriesY = zeros( Float32, numT, numpoints )
        TrajectoriesX = zeros( Float32, numT, numpoints )
        TrajectoriesZ = zeros( Float32, numT, numpoints )

        # Length of the each axis of the vector field. 
        dims  = ( length( 1:size(U,1) ), length( 1:size(U,2) ), length( 1:size(U,3) ) );

        # The user can limit the simulation to a certain subregion of the vector field. 
        sampling_region = [ length( subregion[i] ) == 0 ? (2:dims[i]-1) : subregion[i] for i in 1:3 ]; 

        scale = (typeof(scale)<:Number) ? (scale,scale,scale) : scale

        for pidx in 1:numpoints

            # Placing a new particle inside the vector-field. This is done by 
            # randomly picking a random position withing the vector-field. 
            starting_pos = rand.( sampling_region ); 

            # Recording the starting position in the first timepoints in the trajectory tables for point $pidx. 
            TrajectoriesY[ 1, pidx ] = Float32( starting_pos[1] )
            TrajectoriesX[ 1, pidx ] = Float32( starting_pos[2] )
            TrajectoriesZ[ 1, pidx ] = Float32( starting_pos[3] )

            # Sampling the translation at the current ( starting ) position
            dY = Float32( scale[1] * U[ starting_pos..., T0 ] )
            dX = Float32( scale[2] * V[ starting_pos..., T0 ] )
            dZ = Float32( scale[3] * W[ starting_pos..., T0 ] )

            # moving forward in time, from T0 to T1
            for t in 2:numT

                # New_pos = previous position + translation (dU,dV,dW); 
                updated_pos = ( TrajectoriesY[t-1, pidx], TrajectoriesX[t-1, pidx], TrajectoriesZ[t-1, pidx] ) .+ ( dY, dX, dZ ); 

                # Recording the updated position in the trajectory tables
                TrajectoriesY[t,pidx] = updated_pos[1]
                TrajectoriesX[t,pidx] = updated_pos[2]
                TrajectoriesZ[t,pidx] = updated_pos[3]

                # Obtaining the integer index of the updated position
                int_updated_pos = round.( Int64, updated_pos ); 

                # If the (integer) updated position is out of the coordinates of the vector field, stop
                if any( int_updated_pos .< 1 ) || any( int_updated_pos .> dims ) 
                    TrajectoriesY[ t:end, pidx ] .= TrajectoriesY[ t-1, pidx ]
                    TrajectoriesX[ t:end, pidx ] .= TrajectoriesX[ t-1, pidx ]
                    TrajectoriesZ[ t:end, pidx ] .= TrajectoriesZ[ t-1, pidx ]
                    break
                end

                # Sampling the translation at the (integer) updated position
                dY = Float32( scale[1] * U[ int_updated_pos..., T0+t-1 ] )
                dX = Float32( scale[2] * V[ int_updated_pos..., T0+t-1 ] )
                dZ = Float32( scale[3] * W[ int_updated_pos..., T0+t-1 ] )

            end

        end

        return TrajectoriesY, TrajectoriesX, TrajectoriesZ
    end


    function PIVtrajectories_grid( U::Array{P,4}, V::Array{P,4}, W::Array{P,4}, T0, T1, numpoints; 
                                    subregion=( 1:-1, 1:-1, 1:-1 ), step=(1,1,1), scale=(1,1,1) ) where {P<:Real}

        # Length of the each axis of the vector field. 
        dims  = ( length( 1:size(U,1) ), length( 1:size(U,2) ), length( 1:size(U,3) ) );

        # The user can limit the simulation to a certain subregion of the vector field. 
        sampling_region = [ length( subregion[i] ) == 0 ? (2:step[i]:dims[i]-1) : Base.StepRange( subregion[i].start, step[i], subregion[i].stop ) for i in 1:3 ];
        scale = (typeof(scale)<:Number) ? (scale,scale,scale) : scale

        numT = T1 - T0;
        TrajectoriesY = zeros( Float32, numT, prod( length.(sampling_region) ) )
        TrajectoriesX = zeros( Float32, numT, prod( length.(sampling_region) ) )
        TrajectoriesZ = zeros( Float32, numT, prod( length.(sampling_region) ) )


        pidx = 0; 
        for z in sampling_region[3], x in sampling_region[2], y in sampling_region[1]

            pidx += 1; 

            # Placing a new particle inside the vector-field. This is done by 
            # randomly picking a random position withing the vector-field. 
            starting_pos = (y,x,z); 

            # Recording the starting position in the first timepoints in the trajectory tables for point $pidx. 
            TrajectoriesY[ 1, pidx ] = Float32( starting_pos[1] )
            TrajectoriesX[ 1, pidx ] = Float32( starting_pos[2] )
            TrajectoriesZ[ 1, pidx ] = Float32( starting_pos[3] )

            # Sampling the translation at the current ( starting ) position
            dY = Float32( scale[1] * U[ starting_pos..., T0 ] )
            dX = Float32( scale[2] * V[ starting_pos..., T0 ] )
            dZ = Float32( scale[3] * W[ starting_pos..., T0 ] )

            # moving forward in time, from T0 to T1
            for t in 2:numT

                # New_pos = previous position + translation (dU,dV,dW); 
                updated_pos = ( TrajectoriesY[t-1, pidx], TrajectoriesX[t-1, pidx], TrajectoriesZ[t-1, pidx] ) .+ ( dY, dX, dZ ); 

                # Recording the updated position in the trajectory tables
                TrajectoriesY[t,pidx] = updated_pos[1]
                TrajectoriesX[t,pidx] = updated_pos[2]
                TrajectoriesZ[t,pidx] = updated_pos[3]

                # Obtaining the integer index of the updated position
                int_updated_pos = round.( Int64, updated_pos ); 

                # If the (integer) updated position is out of the coordinates of the vector field, stop
                if any( int_updated_pos .< 1 ) || any( int_updated_pos .> dims ) 
                    TrajectoriesY[ t:end, pidx ] .= TrajectoriesY[ t-1, pidx ]
                    TrajectoriesX[ t:end, pidx ] .= TrajectoriesX[ t-1, pidx ]
                    TrajectoriesZ[ t:end, pidx ] .= TrajectoriesZ[ t-1, pidx ]
                    break
                end

                # Sampling the translation at the (integer) updated position
                dY = Float32( scale[1] * U[ int_updated_pos..., T0+t-1 ] )
                dX = Float32( scale[2] * V[ int_updated_pos..., T0+t-1 ] )
                dZ = Float32( scale[3] * W[ int_updated_pos..., T0+t-1 ] )
            end
        end

        return TrajectoriesY, TrajectoriesX, TrajectoriesZ
    end

end