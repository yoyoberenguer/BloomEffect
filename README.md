# BLOOM
Bloom Effect
IN PROGRESS (will be update very soon)

## Definition:
```
Bloom effet is a computer graphics effect used in video games, demos,
and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.
```
## Requirements:
```
- Pygame 3
- Cython (C extension for python) 
- A C compiler for windows (Visual Studio, MinGW etc) install on your system 
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system, 
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/
```

## Method
```
Acronyme : bpf (Bright Pass Filter)

1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
  bpf24_b_c or bpf32_b_c (adjust the luminence threshold value to get the best filter effect).
  
2) Downside the newly created bpf image into sub-surface downscale factor x2, x4, x8, x16 using 
   pygame transform.scale method. No need to use smoothscale (bilinear filtering method).
  
3) Apply a Gaussian blur 5x5 filter on each of the downsized bpf images (if variable smooth_ is > 1, 
   then the Gaussian filter 5x5 will by applied more than once. Note, this will have little effect 
   on the final image quality. 
  
4) Re-scale to orinial sizes all the bpf images using a bilinear filtering method.
   Note : Using an un-filtered rescaling method will pixelate the final output image.
   Recommandation: For best performances sets smoothscale acceleration.
   A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
   'SSE' allows SSE extensions as well. 
  
5) Blit all the bpf images on the original pygame surface (input image), use pygame additive 
   blend mode effect.
```

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/image1.png)

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/image2.png)
