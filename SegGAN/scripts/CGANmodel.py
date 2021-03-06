# model.netG
DataParallel(                                                                                                [27/1995]
  (module): NetGenerator(
    (model): Sequential(
      (0): Downsample(
        (Downsample): Sequential(
          (0): Conv2d(23, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ReLU(inplace)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (3): ReLU(inplace)
          (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (8): ReLU(inplace)
          (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (15): ReLU(inplace)
        )
      )
      (1): DilatedBlock(
        (conv_block): Sequential(
          (0): ReflectionPad2d((1, 1, 1, 1))
          (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
          (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (3): ReLU(inplace)
        )
      )
      (2): DilatedBlock(
        (conv_block): Sequential(
          (0): ReflectionPad2d((1, 1, 1, 1))
          (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
          (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (3): ReLU(inplace)
        )
      )
      (3): DilatedBlock(
        (conv_block): Sequential(
          (0): ReflectionPad2d((1, 1, 1, 1))
          (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
          (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (3): ReLU(inplace)
        )
      )
      (4): Upsample(
        (Upsample): Sequential(
          (0): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (2): ReLU(inplace)
          (3): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (5): ReLU(inplace)
          (6): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (7): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (8): ReLU(inplace)
          (9): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
          (10): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
          (11): ReLU(inplace)
          (12): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False)
          (14): ReLU(inplace)
          (15): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (16): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False)
          (17): ReLU(inplace)
          (18): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
          (19): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False)
          (20): ReLU(inplace)
          (21): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False)
          (23): ReLU(inplace)
          (24): ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (25): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False)
          (26): ReLU(inplace)
          (27): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (28): Tanh()
        )
      )
    )
  )
)

          
>>> net.netD
DataParallel(
  (module): NLayerDiscriminator(
    (main): Sequential(
      (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): LeakyReLU(0.01)
      (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (3): LeakyReLU(0.01)
      (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (5): LeakyReLU(0.01)
      (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (7): LeakyReLU(0.01)
      (8): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (9): LeakyReLU(0.01)
      (10): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (11): LeakyReLU(0.01)
      (12): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (13): LeakyReLU(0.01)
      (14): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (15): LeakyReLU(0.01)
    )
    (conv1): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (conv2): Conv2d(512, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
