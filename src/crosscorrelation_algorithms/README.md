
quickPIV offers 3 cross-correlation algorithms, which implement different similarity measures for determining the translation that best aligns the two input images:

+ **fftcc ( FFT cross-correlation )**: This is the standard (unnormalized) cross-correlation given by the formula: $ (f\star g)(s) = \int_{-\infty}^{\infty} f(x)g(x+s) dx$. In other words, unnormalized cross-correlation involves computing the dot product between $f$ and $g$ at each possible translation $s$ between $f$ and $g$. I would like to emphasize: *the dot product is the underlying similarity measure in unnormalized cross-correlation*. In general, the dot product between two images will be maximized when bright structures in "image 1" are overlaid with bright structures in "image 2". Thus, the dot product is influenced both by the spatial distribution of the structures, as well as the brightness of the structures in the cross-correlated images. As a consequence, unnormalized cross-correlation has an implicit bias towards translations that overlay "image 1" with regions of bright intensities in "image 2". This limits the applicability of ```fftcc``` to datasets that contain homogeneously bright structures on a homogeneously dark background. <br>
NOTE: as the name suggests, ``fftcc`` uses the Fast Fourier Transform (provided by ```FFTW.jl```) to compute the unnormalized cross-correlation in the frequency domain. <br><br>

+ **zncc ( zero-normalized cross-correlation )**:  This form of cross-correlation is given by the formula: $...$. The result of this normalization is that zncc computes the normalized dot product at each translation, instead of the standard dot product. Therefore, ``zncc`` is equivalent to normalizing both input images (centering them around the mean and dividing the intesities by the total energy of each image), and the translation that maximizes ```zncc``` is only influenced by the spatial distribution of intensities, not by brigthness. <br>
NOTE: In order to 
<br><br>

+ **nsqecc ( normalized squared error cross-correlation )**:This form of cross-correlation arose from wanting to implement a similarity measure that "minimizes differences" between the input images, instead of "maximizing some form of dot product". 



Displacement in quickPIV are extracted from only fully-overlapping translations in the cross-correlation matrix. This applies to all cross-correlation algorithms, starting with FFTC, and extending to ZNCC and NSQECC.