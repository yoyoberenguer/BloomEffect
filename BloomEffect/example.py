"""
Example using bloom_effect24
"""
import time
import os

try:
    import BloomEffect
    from BloomEffect.bloom import bloom_effect24
except ImportError:
    raise ImportError(
        '\n<BloomEffect> library is missing on your system.'
        "\nTry: \n   C:\\pip install BloomEffect on a window command prompt.")

import pygame

PROJECT_PATH = list(BloomEffect.__path__)
os.chdir(PROJECT_PATH[0])

width, height = 256 * 4, 256
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bloom Effect")

image = pygame.image.load('Assets/Aliens.jpg').convert()
image = pygame.transform.smoothscale(image, (256, 256))

bloom_image_128 = bloom_effect24(image, 128)
bloom_image_100 = bloom_effect24(image, 100)
bloom_image_80  = bloom_effect24(image, 80)
bloom_image_20  = bloom_effect24(image, 20)

timer = time.time()
while 1:
    pygame.event.pump()

    screen.blit(bloom_image_128, (0, 0))
    screen.blit(bloom_image_100, (256, 0))
    screen.blit(bloom_image_80, (512, 0))
    screen.blit(bloom_image_20, (768, 0))
    if time.time() - timer > 5:
        pygame.image.save(screen, "Assets/bloom_bpf_values.png")
        break

    pygame.display.flip()

image = pygame.image.load('Assets/i3.png').convert()
image = pygame.transform.smoothscale(image, (256, 256))

bloom_image_128 = bloom_effect24(image, 1, smooth_=1)
bloom_image_100 = bloom_effect24(image, 1, smooth_=5)
bloom_image_80  = bloom_effect24(image, 1, smooth_=8)
bloom_image_20  = bloom_effect24(image, 1, smooth_=10)

screen.fill((0, 0, 0))

timer = time.time()
while 1:
    pygame.event.pump()
    screen.blit(bloom_image_128, (0, 0))
    screen.blit(bloom_image_100, (256, 0))
    screen.blit(bloom_image_80, (512, 0))
    screen.blit(bloom_image_20, (768, 0))
    if time.time() - timer > 5:
        pygame.image.save(screen, "Assets/bloom_smooth_values.png")
        break

    pygame.display.flip()

width, height = 256 * 2, 256
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("bloom effect")

image = pygame.image.load('Assets/I1.png').convert()
image = pygame.transform.smoothscale(image, (256, 256))

bloom_image = bloom_effect24(image, 58, smooth_=10)

screen.fill((0, 0, 0))

timer = time.time()
while 1:
    pygame.event.pump()
    screen.blit(image, (0, 0))
    screen.blit(bloom_image, (256, 0))
    if time.time() - timer > 5:
        pygame.image.save(screen, "Assets/i3_bloom.png")
        break

    pygame.display.flip()

image = pygame.image.load('Assets/i2.png').convert()
image = pygame.transform.smoothscale(image, (256, 256))

bloom_image = bloom_effect24(image, 58, smooth_=10)

screen.fill((0, 0, 0))

timer = time.time()
while 1:
    pygame.event.pump()
    screen.blit(image, (0, 0))
    screen.blit(bloom_image, (256, 0))
    if time.time() - timer > 5:
        pygame.image.save(screen, "Assets/i2_bloom.png")
        break

    pygame.display.flip()

image = pygame.image.load('Assets/control.png').convert()
image = pygame.transform.smoothscale(image, (256, 256))

bloom_image = bloom_effect24(image, 80, smooth_=8)

screen.fill((0, 0, 0))

timer = time.time()
while 1:
    pygame.event.pump()
    screen.blit(image, (0, 0))
    screen.blit(bloom_image, (256, 0))
    if time.time() - timer > 5:
        pygame.image.save(screen, "Assets/text_bloom.png")
        break

    pygame.display.flip()
pygame.quit()
