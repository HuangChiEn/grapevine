# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from .mlpmixer import mixer_s32_224 as default_mixer_small
from .mlpmixer import mixer_b32_224 as default_mixer_base
from .mlpmixer import mixer_l32_224 as default_mixer_large


def mlpmixer_small(method, *args, **kwargs):
    return default_mixer_small(*args, **kwargs)


def mlpmixer_base(method, *args, **kwargs):
    return default_mixer_base(*args, **kwargs)


def mlpmixer_large(method, *args, **kwargs):
    return default_mixer_large(*args, **kwargs)


__all__ = ["mlpmixer_small", "mlpmixer_base", "mlpmixer_large"]
