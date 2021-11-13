"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import timeit
import unittest


# PYGAME IS REQUIRED
import numpy

try:
    import pygame

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, make_surface
from pygame.image import frombuffer

try:
    import BloomEffect
except ImportError:
    raise ImportError('\n<BloomEffect> library is missing on your system.'
                      "\nTry: \n   C:\\pip install BloomEffect on a window command prompt.")

from BloomEffect.bloom import blur5x5_array24, blur5x5_array32, blur5x5_array24_inplace, blur5x5_array32_inplace, \
    test_bpf24_c, test_bpf24_inplace, test_bpf32_c, test_bpf32_inplace, kernel_deviation, filtering24,\
    build_mask_from_surface, filtering32, bloom_effect_array24_inplace, bloom_effect_array32_inplace, \
    bloom_effect_array24, bloom_effect_array32


import time
import os

PROJECT_PATH = BloomEffect.__path__
os.chdir(PROJECT_PATH[0] + "\\tests")


class TestBlur5x5Array24(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("Blur5x5Array24")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background = pygame.transform.smoothscale(background, (800, 600))

        # Basic checks blurring an array with only 255 values
        # Full array with 255 values
        full_255 = numpy.full((800, 600, 3), (255, 255, 255), numpy.uint8)
        blur_array = blur5x5_array24(full_255)
        self.assertTrue(numpy.array_equal(blur_array, full_255))

        # Basic checks blurring an array with only 0 values
        full_0 = numpy.full((800, 600, 3), (0, 0, 0), numpy.uint8)
        blur_array = blur5x5_array24(full_0)
        self.assertTrue(numpy.array_equal(blur_array, full_0))

        self.assertEqual(background.get_bytesize(), 4)
        rgb_array = array3d(background)
        blur_rgb_array = blur5x5_array24(rgb_array)
        self.assertIsInstance(blur_rgb_array, numpy.ndarray)

        background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        rgb_array = array3d(background)
        blur_rgb_array = blur5x5_array24(rgb_array)
        w, h, d = blur_rgb_array.shape

        # Check if the array kept the same dimensions width and height
        # Check also that the alpha channel as been removed
        self.assertTrue(d == 3)
        self.assertTrue(w == 800)
        self.assertTrue(h == 600)

        self.assertIsInstance(blur_rgb_array, numpy.ndarray)
        blur_surface = make_surface(blur_rgb_array)

        # Array shape (w, h, 3) uint8
        rgb_array = numpy.zeros((800, 600, 3), numpy.uint8)
        blur5x5_array24(rgb_array)
        # Array shape (w, h, 4) uint8
        rgb_array = numpy.zeros((800, 600, 4), numpy.uint8)
        blur5x5_array24(rgb_array)

        # Testing wrong datatype
        rgb_array = numpy.zeros((800, 600, 3), numpy.float32)
        self.assertRaises(ValueError, blur5x5_array24, rgb_array)
        # Testing wrong datatype
        rgb_array = numpy.zeros((800, 600, 3), numpy.int8)
        self.assertRaises(ValueError, blur5x5_array24, rgb_array)

        display(screen, checker, background, blur_surface)


class TestBlur5x5Array32(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("Blur5x5Array32")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        # Basic checks blurring an array with only 255 values
        # Full array with 255 values
        full_255 = numpy.full((800, 600, 4), (255, 255, 255, 0), numpy.uint8)
        blur_array = blur5x5_array32(full_255)
        self.assertTrue(numpy.array_equal(blur_array, full_255))

        # Basic checks blurring an array with only 0 values
        full_0 = numpy.full((800, 600, 4), (0, 0, 0, 0), numpy.uint8)
        blur_array = blur5x5_array32(full_0)
        self.assertTrue(numpy.array_equal(blur_array, full_0))

        self.assertEqual(background.get_bytesize(), 4)
        rgba_array = array3d(background)
        blur_rgba_array = blur5x5_array32(rgba_array)
        self.assertIsInstance(blur_rgba_array, numpy.ndarray)

        background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        alpha_array = array_alpha(background)
        rgb_array   = array3d(background)
        rgba_array  = numpy.dstack((rgb_array, alpha_array)).transpose(1, 0, 2)
        blur_rgba_array = blur5x5_array32(rgba_array)
        w, h, d = blur_rgba_array.shape

        # Check if the array kept the same dimensions width and height
        # Check also that the alpha channel as been removed
        self.assertTrue(d == 4)
        self.assertTrue(w == 600)
        self.assertTrue(h == 800)

        self.assertIsInstance(blur_rgba_array, numpy.ndarray)
        blur_surface = pygame.image.frombuffer(blur_rgba_array, (h, w), 'RGBA')

        # Array shape (w, h, 4) uint8
        rgba_array = numpy.zeros((800, 600, 4), numpy.uint8)
        blur5x5_array32(rgba_array)

        # Testing wrong datatype
        rgba_array = numpy.zeros((800, 600, 4), numpy.float32)
        self.assertRaises(ValueError, blur5x5_array32, rgba_array)
        # Testing wrong datatype
        rgba_array = numpy.zeros((800, 600, 4), numpy.int8)
        self.assertRaises(ValueError, blur5x5_array32, rgba_array)
        display(screen, checker, background, blur_surface)


class TestBlur5x5Array24Inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("Blur5x5Array24Inplace")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()
        blur5x5_array24_inplace(pixels3d(background))

        # Testing wrong datatype
        rgba_array = numpy.zeros((800, 600, 4), numpy.float32)
        self.assertRaises(ValueError, blur5x5_array32, rgba_array)
        # Testing wrong size (depth = 4)
        rgba_array = numpy.zeros((800, 600, 4), numpy.int8)
        self.assertRaises(ValueError, blur5x5_array32, rgba_array)

        display(screen, checker, background_cp, background)



class TestBlur5x5Array32Inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("Blur5x5Array32Inplace")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        # blur5x5_array32_inplace(pixels3d(background))

        # # Testing wrong datatype
        # rgba_array = numpy.zeros((800, 600, 4), numpy.float32)
        # self.assertRaises(ValueError, blur5x5_array32, rgba_array)
        # # Testing wrong size (depth = 3)
        # rgba_array = numpy.zeros((800, 600, 3), numpy.int8)
        # self.assertRaises(ValueError, blur5x5_array32, rgba_array)

        # timer = time.time()
        # while 1:
        #     pygame.event.pump()
        #
        #     screen.blit(checker, (0, 0))
        #     screen.blit(background_cp, (0, 0))
        #     screen.blit(background, (800, 0))
        #
        #     if time.time() - timer > 5:
        #         break
        #
        #     pygame.display.flip()


class Testbpf24_c(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("bright pass filter (bpf24)")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        # array3d
        bpf_surface = test_bpf24_c(array3d(background), 40)
        self.assertRaises(OverflowError, test_bpf24_c, array3d(background), -40)
        self.assertRaises(OverflowError, test_bpf24_c, array3d(background), 440)

        # pixel3d
        bpf_surface = test_bpf24_c(pixels3d(background), 40)

        # Test a single pixel
        bpf_array = array3d(bpf_surface)
        bck_array = array3d(background)

        r = bck_array[100, 100, 0]
        g = bck_array[100, 100, 1]
        b = bck_array[100, 100, 2]

        lum = r * 0.299 + g * 0.587 + b * 0.114
        # no div by zero lum must be strictly > 0
        if lum > 40:
            c = (lum - 40) / lum
            self.assertEqual(bpf_array[100, 100, 0], int(r * c))
            self.assertEqual(bpf_array[100, 100, 1], int(g * c))
            self.assertEqual(bpf_array[100, 100, 2], int(b * c))

        display(screen, checker, background, bpf_surface)



class Testbpf24_inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("bright pass filter (bpf24_inplace)")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        # array3d
        # test_bpf24_inplace(array3d(background), 40)

        # pixel3d
        test_bpf24_inplace(pixels3d(background), 45)

        # test argument threshold
        self.assertRaises(OverflowError, test_bpf24_inplace, array3d(background), -40)
        self.assertRaises(OverflowError, test_bpf24_inplace, array3d(background), 440)

        display(screen, checker, background_cp, background)



class Testbpf32_c(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("bright pass filter (bpf32)")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        # array3d
        bpf_array = test_bpf32_c(background, 40)
        self.assertRaises(OverflowError, test_bpf32_c, background, -40)
        self.assertRaises(OverflowError, test_bpf32_c, background, 440)

        # pixel3d
        bpf_array = test_bpf32_c(background, 40)
        bpf_surface = pygame.image.frombuffer(
            numpy.ascontiguousarray(bpf_array.transpose(1, 0, 2)), (800, 600), 'RGBA')
        # Test a single pixel
        bck_array = array3d(background)

        r = bck_array[100, 100, 0]
        g = bck_array[100, 100, 1]
        b = bck_array[100, 100, 2]

        lum = r * 0.299 + g * 0.587 + b * 0.114
        # no div by zero lum must be strictly > 0
        if lum > 40:
            c = (lum - 40) / lum
            self.assertEqual(bpf_array[100, 100, 0], int(r * c))
            self.assertEqual(bpf_array[100, 100, 1], int(g * c))
            self.assertEqual(bpf_array[100, 100, 2], int(b * c))

        display(screen, checker, background, bpf_surface)


class Testbpf32_inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))
        pygame.display.set_caption("bright pass filter (bpf32_inplace)")
        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        self.assertRaises(OverflowError, test_bpf32_inplace, background, -40)
        self.assertRaises(OverflowError, test_bpf32_inplace, background, 440)

        # pixel3d
        test_bpf32_inplace(background, 45)

        display(screen, checker, background_cp, background)


class Test_kernel_deviation(unittest.TestCase):

    def runTest(self) -> None:
        kernel = numpy.array([
            0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067,
            0.00002292,	0.00078633,	0.00655965,	0.01330373,	0.00655965,	0.00078633,	0.00002292,
            0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
            0.00038771,	0.01330373,	0.11098164,	0.22508352,	0.11098164,	0.01330373,	0.00038771,
            0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
            0.00002292,	0.00078633,	0.00655965,	0.01330373,	0.00655965,	0.00078633,	0.00002292,
            0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067
        ])

        kernel = kernel.reshape(7, 7)
        kernel7x7 = kernel_deviation(sigma = 0.84089642, kernel_size=7)
        self.assertTrue(numpy.array_equal(numpy.around(kernel, 5),  numpy.around(kernel7x7, decimals=5)))
        kernel7x7 = numpy.around(kernel7x7, decimals=8)

        for i in range(7):
            for j in range(7):
                if round(kernel[i, j], 5) != round(kernel7x7[i, j], 5):
                    print("\n ", kernel[i, j], kernel7x7[i, j])
                    break


def display(screen, checker, image1, image2):
    timer = time.time()
    while 1:
        pygame.event.pump()
        screen.blit(checker, (0, 0))
        screen.blit(image1, (0, 0))
        screen.blit(image2, (800, 0))

        if time.time() - timer > 5:
            break

        pygame.display.flip()


class TestFiltering24(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("filter24_inplace")

        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        mask_image = pygame.image.load('../Assets/color_mask_circle.png').convert()
        mask_image = pygame.transform.smoothscale(mask_image, (800, 600))

        mask_array = build_mask_from_surface(mask_image, invert_mask=False)
        filtering24(background, mask_array)

        display(screen, checker, background_cp, background)

        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()
        mask_array = build_mask_from_surface(background, invert_mask=True)
        filtering24(background, mask_array)
        pygame.display.set_caption("filter24_inplace with mask invert_mask=True")

        display(screen, checker, background_cp, background)

        # Array filled with 1.0
        test_mask = numpy.full((800, 600), 1.0, numpy.float32)
        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        filtering24(background, test_mask)

        arr1 = array3d(background)
        arr2 = array3d(background_cp)
        self.assertTrue(numpy.array_equal(arr1, arr2))
        # display(screen, checker, background_cp, background)

        # Array filled with 0.0
        test_mask = numpy.full((800, 600), 0.0, numpy.float32)
        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        filtering24(background, test_mask)
        arr1 = array3d(background)
        self.assertTrue(numpy.sum(arr1) == 0)
        # display(screen, checker, background_cp, background)


class TestFiltering32(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("filter32_inplace")

        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        mask_image = pygame.image.load('../Assets/color_mask_circle.png').convert_alpha()
        mask_image = pygame.transform.smoothscale(mask_image, (800, 600))

        mask_array = build_mask_from_surface(mask_image, invert_mask=False)
        filtering32(background, mask_array)

        display(screen, checker, background_cp, background)

        background = pygame.image.load('../Assets/I2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()
        mask_array = build_mask_from_surface(background, invert_mask=True)
        filtering32(background, mask_array)
        pygame.display.set_caption("filter32 with mask invert_mask=True")

        display(screen, checker, background_cp, background)

        # Array filled with 1.0
        test_mask = numpy.full((800, 600), 1.0, numpy.float32)
        background = pygame.image.load('../Assets/I2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        filtering32(background, test_mask)

        arr1 = array3d(background)
        arr2 = array3d(background_cp)
        self.assertTrue(numpy.array_equal(arr1, arr2))
        # display(screen, checker, background_cp, background)

        # Array filled with 0.0
        test_mask = numpy.full((800, 600), 0.0, numpy.float32)
        background = pygame.image.load('../Assets/I2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        filtering32(background, test_mask)
        arr1 = array3d(background)
        self.assertTrue(numpy.sum(arr1) == 0)
        # display(screen, checker, background_cp, background)


class TestBloomEffectArray24Inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("Test Bloom effect array24 Inplace")

        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        bloom_effect_array24_inplace(background, 128, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array24_inplace, background, -5, 1)
        self.assertRaises(AssertionError, bloom_effect_array24_inplace, background, 1500, 1)
        self.assertRaises(AssertionError, bloom_effect_array24_inplace, background, 255, -8)
        self.assertRaises(AssertionError, bloom_effect_array24_inplace, 1, 255, True)

        display(screen, checker, background_cp, background)

        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        bloom_effect_array24_inplace(background, 128, fast_=True)

        display(screen, checker, background_cp, background)


class TestBloomEffectArray32Inplace(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("Test Bloom effect array32 Inplace")

        checker = pygame.image.load('../Assets/background_checker.png').convert_alpha()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/I2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        bloom_effect_array32_inplace(background, 128, fast_=False)

        self.assertRaises(AssertionError, bloom_effect_array32_inplace, background, -5, 1)
        self.assertRaises(AssertionError, bloom_effect_array32_inplace, background, 1500, 1)
        self.assertRaises(AssertionError, bloom_effect_array32_inplace, background, 255, -8)
        self.assertRaises(AssertionError, bloom_effect_array32_inplace, 1, 255, True)

        display(screen, checker, background_cp, background)

        background = pygame.image.load('../Assets/I2.png').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        bloom_effect_array32_inplace(background, 128, fast_=True)

        display(screen, checker, background_cp, background)


class TestBloomEffectArray24(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("Test Bloom effect array24")

        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background = pygame.transform.smoothscale(background, (800, 600))

        image = bloom_effect_array24(background, threshold_=45, smooth_=1, fast_=False)

        display(screen, checker, background, image)
        self.assertRaises(AssertionError, bloom_effect_array24, background, -5, 1, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array24, background, 258, 1, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array24, background, 45, -5, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array24, background, 45, 1, fast_=-8)
        self.assertRaises(AssertionError, bloom_effect_array24, 1, 1, fast_=False)

        self.assertIsInstance(image, pygame.Surface)

        # test with convert_alpha
        background = pygame.image.load('../Assets/i2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        image = bloom_effect_array24(background, threshold_=45, smooth_=1, fast_=True)
        display(screen, checker, background, image)

        # test smooth > 1
        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        image = bloom_effect_array24(background, threshold_=45, smooth_=15, fast_=True)
        display(screen, checker, background, image)


class TestBloomEffectArray32(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 600
        screen = pygame.display.set_mode((width * 2, height))

        pygame.display.set_caption("Test Bloom effect array32")

        checker = pygame.image.load('../Assets/background_checker.png').convert()
        checker = pygame.transform.smoothscale(checker, (1600, 600))

        background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))

        image = bloom_effect_array32(background, threshold_=45, smooth_=1, fast_=False)

        display(screen, checker, background, image)
        self.assertRaises(AssertionError, bloom_effect_array32, background, -5, 1, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array32, background, 258, 1, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array32, background, 45, -5, fast_=False)
        self.assertRaises(AssertionError, bloom_effect_array32, background, 45, 1, fast_=-8)
        self.assertRaises(AssertionError, bloom_effect_array32, 1, 1, fast_=False)

        self.assertIsInstance(image, pygame.Surface)

        # test with convert_alpha
        background = pygame.image.load('../Assets/i2.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        image = bloom_effect_array32(background, threshold_=45, smooth_=1, fast_=True)
        display(screen, checker, background, image)

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        self.assertRaises(ValueError, bloom_effect_array32, background, threshold_=45, smooth_=15, fast_=True)

        background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        image = bloom_effect_array32(background, threshold_=45, smooth_=15, fast_=True)
        display(screen, checker, background, image)

def run_testsuite():

    suite = unittest.TestSuite()

    suite.addTests([
                    TestBlur5x5Array24(),
                    TestBlur5x5Array32(),
                    TestBlur5x5Array24Inplace(),
                    TestBlur5x5Array32Inplace(),
                    Testbpf24_c(),
                    Testbpf24_inplace(),
                    Testbpf32_c(),
                    Testbpf32_inplace(),
                    Test_kernel_deviation(),
                    TestFiltering24(),
                    TestFiltering32(),
                    TestBloomEffectArray24Inplace(),
                    TestBloomEffectArray32Inplace(),
                    TestBloomEffectArray24(),
                    TestBloomEffectArray32()
                    ])

    unittest.TextTestRunner().run(suite)
    pygame.quit()



if __name__ == '__main__':
    run_testsuite()
    pygame.quit()

