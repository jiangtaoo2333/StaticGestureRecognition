'''
@Author: Jiangtao
@Date: 2020-03-24 09:09:46
@LastEditors: Jiangtao
@LastEditTime: 2020-03-24 10:30:45
@Description: 
'''
#
#   Extra lightnet layers
#   Copyright EAVISE
#
"""
.. Note::
   Every parameter that can get an int or tuple will behave as follows. |br|
   If a tuple of 2 ints is given, the first int is used for the height and the second for the width. |br|
   If an int is given, both the width and height are set to this value.
"""

from ._basic_layer import *
from .reuse_basic_layer import *
from .gather_layer import *

from .am_softmax import AMSoftmax
from .arc_softmax import ArcSoftmax
from .circle_softmax import CircleSoftmax