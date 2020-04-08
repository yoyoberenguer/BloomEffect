# BLOOM
Bloom Effect using Cython

Left image with bloom effect 

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/image1.png)

Left image with bloom effect

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/image2.png)


## How to spicy up your text or demo
```
Achieve with 30 passes of blur --> smooth_=30
image = bloom_effect_buffer24(im, j, smooth_=30)
```
![alt text](https://github.com/yoyoberenguer/bloom/blob/master/Screendump13427.png)


## Definition:
```
Bloom effet is a computer graphics effect used in video games, demos,
and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.
```
## Requirements:
```
- Pygame 3
- Numpy 
- Cython (C extension for python) 
- A C compiler for windows (Visual Studio, MinGW etc) install on your system 
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system, 
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/
```

## Method

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/BLOOM.png)

```
Acronyme : bpf (Bright Pass Filter)

1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
  bpf24_b_c or bpf32_b_c (adjust the luminence threshold value to get the best filter effect).
  
2) Downscale the newly created bpf image into sub-surface factor x2, x4, x8, x16 using 
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

## Building the project 
```
In a command prompt and under the directory containing the source files
C:\>python setup_bloom.py build_ext --inplace 

If the compilation fail, refers to the requirement section and make sure cython 
and a C-compiler are correctly install on your system. 

```
## Two methods
```
# Below bloom method is using massively numpy.ndarray to manipulate data.
bloom = bloom_effect_array(surface, threshold, smooth_=1)

# Below method is using BufferProxy or C-Buffer data structure. 
# It is also the fastest algorithm available.
bloom = bloom_effect_buffer(surface, threshold, smooth_=1)

Surface   : pygame.Surface to be bloom (compatible 24, 32-bit format) 
threshold : Integer value for the bright pass filter, filter threshold value
smooth_   : Smooth define the quantity of Gaussian blur5x5 kernel passes that will be 
            applied to all sub-surface (default is 1, vertical & horizontal)
            Note the Gaussian algorithm is cpu demanding. 4 is plenty smoothing
```
Left image with smooth_=1, 
right image with smooth_=10

![alt text](https://github.com/yoyoberenguer/bloom/blob/master/image3.png)


## Tips
```
C:\Users\user\Downloads\BLOOM-master>python test_bloom.py

If you get the following error message after execution, refer to 
the section building the project.

<<ModuleNotFoundError: No module named 'bloom'>>

```

## Example in python IDE 
```

import pygame
import numpy
from numpy import asarray, uint8, float32, zeros, float64

#<---  HERE --->
import bloom
from bloom import bloom_effect_buffer, bloom_effect_array

im = pygame.image.load("I2.jpg")
im = pygame.transform.smoothscale(im, (600, 600))

w, h = im.get_size()
screen = pygame.display.set_mode((w, h))

i = 0
j = 255
STOP_DEMO = True
while STOP_DEMO:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    # screen.fill((10, 10, 10, 255))
    org = bloom_effect_array(im, j, smooth_=1)
    screen.blit(org, (0, 0))
    pygame.display.flip()

    if keys[pygame.K_ESCAPE]:
        STOP_DEMO = False

    if keys[pygame.K_F8]:
        pygame.image.save(screen, 'Screendump' + str(i) + '.png')
    i += 1
    j -= 1
    if j < 0:
        j = 255
```

## DEMO version
```
Decompress the archive in the source diretory. 
In order to work, demo.exe must be under the same directory with i2.jpg 
Copy/move i2.jpg if necessary to demo.exe location.
```

## Timings
```
print(timeit.timeit("bloom_effect_array(im, 255, smooth_=1)",
                    "from __main__ import bloom_effect_array, im", number=10) / 10)
print(timeit.timeit("bloom_effect_buffer(im, 255, smooth_=1)",
                     "from __main__ import bloom_effect_buffer, im", number=10) / 10)
Method 1:
bloom_effect_array 
texture 600x600 24-bit gives a modest 0.0793 (79ms) processing time

Method 2:
bloom_effect_buffer
texture 600x600 24-bit gives a modest 0.04516 (45ms) processing time

Those values are not too bad considering that all the texture processing is done 
entirely by the CPU.
Soon I will implement a mask method that will improve efficieny of both techniques.
```

### Links
https://learnopengl.com/Advanced-Lighting/Bloom

https://kalogirou.net/2006/05/20/how-to-do-good-bloom-for-hdr-rendering/

https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/

