# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


class Palette(object):
    def __init__(self, bg, text):
        self.bg = bg
        self.text = text
        assert len(self.bg) == len(self.text)

    def __getitem__(self, i):
        return self.bg[i], self.text[i]

    def __len__(self):
        return len(self.bg)


# For some ideas
# https://coolors.co/palettes/trending

Green4Red1 = Palette(
        bg=[
            (56, 102, 65),
            (106, 153, 78),
            (167, 201, 87),
            (242, 232, 207),
            (188, 71, 73)
            ],
        text=[
            (255,255,255),
            (255,255,255),
            (0,0,0),
            (0,0,0),
            (255,255,255),
            ])

PaleContrastToRed = Palette(
        bg=[
            (221, 28, 26),
            (168, 213, 226),
            (249, 166, 32),
            (255, 212, 73),
            (84, 140, 47),
            (16, 73, 17),
            ],
        text=[
            (255, 255, 255),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (255, 255, 255),
            (255, 255, 255),
            ])

UmberToScarlet = Palette(
        bg=[
            (4, 42, 43),
            (94, 177, 191),
            (205, 237, 246),
            (239, 123, 69),
            (216, 71, 39),
            ],
        text=[
            (255, 255, 255),
            (255, 255, 255),
            (0, 0, 0),
            (255, 255, 255),
            (255, 255, 255),
            ])
