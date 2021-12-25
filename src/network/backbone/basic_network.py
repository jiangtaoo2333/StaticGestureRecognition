'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-10-08 13:49:42
@Description: 
'''
import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer

class multi_out_5(nn.Module):
    
    def __init__(self,):
        super(multi_out_5, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone()
        self.mouthBone = layer.mouthBone()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        


        return [stage1,stage2,stage3,stage4,stage5]


class multi_out_5_multiscale(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiscale, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_5_multiscale_WITH_BG(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiscale_WITH_BG, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone_WITH_BG()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]
        
class multi_out_5_PFLD(nn.Module):
    
    def __init__(self,):
        super(multi_out_5_PFLD, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone()
        self.mouthBone = layer.mouthBone()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()
        self.angleBone = layer.angleBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class multi_out_5_elementwise(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_elementwise, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_elementwise()
        self.mouthBone = layer.mouthBone_elementwise()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_5_multiLabel(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiLabel, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.eyeBone_multiscale()
        self.faceBone = layer.faceBone_multiLabel()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_5_multiscale(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiscale, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_5_multiscale_new(nn.Module):

    def __init__(self,):
        
        super(multi_out_5_multiscale_new, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()

    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_5_multiScale_blurpool(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiScale_blurpool, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone_blurpool()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_2(nn.Module):

    def __init__(self,):
        
        super(multi_out_2, self).__init__()

        self.baseBone = layer.backBone()
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()

    def forward(self,x):

        stage0 = self.baseBone(x)
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        
        return [stage1,stage2]

class multi_out_HeatMap(nn.Module):

    def __init__(self,):
        
        super(multi_out_HeatMap, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeHeatmapBone = layer.eyeHeatmap()
        self.mouthHeatmapBone = layer.mouthHeatmap()

        self.eyeBone = layer.eyeBone_multiscale_heatmap()
        self.mouthBone = layer.mouthBone_multiscale_heatmap()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeHeatmapBone(stage0)
        stage2 = self.mouthHeatmapBone(stage0)

        stage3 = self.eyeBone(stage0,stage1)
        stage4 = self.mouthBone(stage0,stage2)

        return [stage1,stage2,stage3,stage4]

class multi_out_5_multiscale_SSD(nn.Module):

    
    def __init__(self,):
        
        super(multi_out_5_multiscale_SSD, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_SSD()
        self.emotionBone = layer.emotionBone_WITH_BG()

    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_8(nn.Module):

    def __init__(self,):
        
        super(multi_out_8, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()

        self.YawBone = layer.YawBone()
        self.PitchBone = layer.PitchBone()
        self.RollBone = layer.RollBone()


    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        stage6 = self.YawBone(stage0)
        stage7 = self.PitchBone(stage0)
        stage8 = self.RollBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_5_multiscale_new_new(nn.Module):

    def __init__(self,):
        
        super(multi_out_5_multiscale_new_new, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        
        self.detectBone = layer.detectBone_multiScale_new()
        self.emotionBone = layer.emotionBone_WITH_BG()

    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_8_new(nn.Module):

    def __init__(self,):
        
        super(multi_out_8_new, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale_new()
        self.emotionBone = layer.emotionBone_WITH_BG()

        self.YawBone = layer.YawBone()
        self.PitchBone = layer.PitchBone()
        self.RollBone = layer.RollBone()


    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        stage6 = self.YawBone(stage0)
        stage7 = self.PitchBone(stage0)
        stage8 = self.RollBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_9(nn.Module):

    def __init__(self,):
        
        super(multi_out_9, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.binaryBone = layer.BinaryBone()

        self.YawBone = layer.YawBone()
        self.PitchBone = layer.PitchBone()
        self.RollBone = layer.RollBone()


    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        stage6 = self.binaryBone(stage0)

        stage7 = self.YawBone(stage0)
        stage8 = self.PitchBone(stage0)
        stage9 = self.RollBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9]

class multi_out_2_3channel(nn.Module):
    
    def __init__(self,):
        super(multi_out_2_3channel, self).__init__()

        self.baseBone = layer.backBone_3channel()
        
        self.eyeBone = layer.eyeBone()
        self.mouthBone = layer.mouthBone()


    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)

        return [stage1,stage2]

class multi_out_5_multiscale_3channel(nn.Module):
    
    def __init__(self,):
        
        super(multi_out_5_multiscale_3channel, self).__init__()

        self.baseBone = layer.backBone_3channel()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone()
        self.emotionBone = layer.emotionBone()

    def forward(self,x):
        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_6(nn.Module):

    def __init__(self,):
        
        super(multi_out_6, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.FaceAreaBone = layer.FaceAreaBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.FaceAreaBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class multi_yolo(nn.Module):

    def __init__(self,):
        
        super(multi_yolo, self).__init__()

        self.baseBone = layer.backBone()
        self.detectBone = layer.detectBone_YOLO()


    def forward(self,x):

        stage0 = self.baseBone(x)
        stage1 = self.detectBone(stage0)

        return stage1

class multi_out_5_HeatMap(nn.Module):

    def __init__(self,):
        
        super(multi_out_5_HeatMap, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_HeatMap()
        self.emotionBone = layer.emotionBone_WITH_BG()

    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        return [stage1,stage2,stage3,stage4,stage5]

class multi_out_6_Binary_HeatMap(nn.Module):

    def __init__(self,):
        
        super(multi_out_6_Binary_HeatMap, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_HeatMap()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.FaceAreaBone = layer.FaceAreaBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.FaceAreaBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class multi_out_6_Binary(nn.Module):

    def __init__(self,):
        
        super(multi_out_6_Binary, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.binaryBone = layer.BinaryBone()

    def forward(self,x):

        
        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        
        stage6 = self.binaryBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class single_out_facearea(nn.Module):

    def __init__(self,):
        
        super(single_out_facearea, self).__init__()

        self.FaceArea = layer.FaceAreaBone_new()
        
    def forward(self,x):

        out = self.FaceArea(x)
        
        return out

class multi_out_6_FaceArea(nn.Module):

    def __init__(self,):
        
        super(multi_out_6_FaceArea, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        # self.FaceAreaBone = layer.FaceAreaBone_five()
        self.FaceAreaBone = layer.FaceAreaBone_seven()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.FaceAreaBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class multi_out_6_Angle(nn.Module):

    def __init__(self,):
        
        super(multi_out_6_Angle, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6]

class multi_out_7(nn.Module):

    def __init__(self,):
        
        super(multi_out_7, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone()
        self.FaceAreaBone = layer.FaceAreaBone_seven()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7]

class multi_out_8(nn.Module):

    def __init__(self,):
        
        super(multi_out_8, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone()
        self.FaceAreaBone = layer.FaceAreaBone_seven()
        self.recogBone = layer.RecognitionBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_8_angle_new(nn.Module):

    def __init__(self,):
        
        super(multi_out_8_angle_new, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven()
        self.recogBone = layer.RecognitionBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_7_0911(nn.Module):

    def __init__(self,):

        super(multi_out_7_0911, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7]

class multi_out_8_20210607(nn.Module):

    def __init__(self,):

        super(multi_out_8_20210607, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.alignQualityBone = layer.alignQualityBone()
    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.alignQualityBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_8_20210426(nn.Module):

    def __init__(self,):

        super(multi_out_8_20210426, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class singleGazeBone(nn.Module):

    def __init__(self,):

        super(singleGazeBone, self).__init__()

        self.baseBone = layer.backBone()
        self.gazeBone = layer.GazeBone()

    def forward(self,x):

        stage0 = self.baseBone(x)
        stage1 = self.gazeBone(stage0)

        return [stage1]

class singleGazeBone_5(nn.Module):

    def __init__(self,):

        super(singleGazeBone_5, self).__init__()

        self.baseBone = layer.backBone()
        self.gazeBone = layer.GazeBone_5()

    def forward(self,x):

        stage0 = self.baseBone(x)
        stage1 = self.gazeBone(stage0)

        return [stage1]

class multi_out_5_0517(nn.Module):
    def __init__(self,):        
        super(multi_out_5_0517, self).__init__()
        # self.baseBone = layer.backBone()

        # self.eyeBone = layer.eyeBone_multiscale_1111()
        # self.mouthBone = layer.mouthBone_multiscale()
        # self.faceBone = layer.faceBone()
        # self.detectBone = layer.detectBone_multiScale()
        # self.emotionBone = layer.emotionBone_WITH_BG()
        # self.angleBone = layer.angleBone_six()
        # self.FaceAreaBone = layer.FaceAreaBone_multiscale()
        # self.recogBone = layer.RecognitionBone()
        # self.FaceCls = layer.KLOccCls()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.FaceCls = layer.KLOccCls()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        # stage8 = self.recogBone(stage0,idTargets)
        stage8 = self.FaceCls(stage0)

        # return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8, stage9]
        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]


class multi_out_9_20210519(nn.Module):

    def __init__(self,):

        super(multi_out_9_20210519, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.gazeBone = layer.GazeBone()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.gazeBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9]

class multi_out_9_20210525(nn.Module):

    def __init__(self,):

        super(multi_out_9_20210525, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.angleRegBone = layer.AngleBoneReg()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.angleRegBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9]


class multi_out_8_1010(nn.Module):

    def __init__(self,):
        
        super(multi_out_8_1010, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_8_1111(nn.Module):

    def __init__(self,):
        
        super(multi_out_8_1111, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale_1111()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class multi_out_9_0408(nn.Module):

    def __init__(self,):
        
        super(multi_out_9_0408, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale_1111()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()
        self.FaceQuality = layer.QualityCls()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)
        stage9 = self.FaceQuality(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8, stage9]

class multi_out_8_0104(nn.Module):

    def __init__(self,):
        
        super(multi_out_8_0104, self).__init__()

        self.baseBone = layer.backBone0104()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]

class Onet(nn.Module):

    def __init__(self,):
        super(Onet, self ).__init__() 
        self.baseBone = layer.eyeBone_Onet()

    def forward(self,x):
        x = self.baseBone(x)
        return [x]

class OnetV2(nn.Module):

    def __init__(self,):
        super(OnetV2, self ).__init__() 
        self.baseBone = layer.eyeBone_OnetV2()

    def forward(self,x):
        x = self.baseBone(x)
        return [x]

class OnetV3(nn.Module):
    
    def __init__(self,):
        super(OnetV3,self).__init__()
        self.baseBone = layer.baseBone_Onet()
        self.alignBone = layer.alignBone_Onet()
        self.binaryFaceBone = layer.binaryBone_Onet()

    def forward(self,x):
        x = self.baseBone(x)
        x1 = self.alignBone(x)
        x2 = self.binaryFaceBone(x)
        return [x1,x2]

class OnetV4(nn.Module):

    def __init__(self,):
        super(OnetV4,self).__init__()
        self.baseBone = layer.baseBone_Onet()
        self.alignBone = layer.alignBone_Onet()
        self.binaryFaceBone = layer.binaryBone_Onet()
        self.gazeBone = layer.gazeBone_Onet()

    def forward(self,x):
        x = self.baseBone(x)
        x1 = self.alignBone(x)
        x2 = self.binaryFaceBone(x)
        x3 = self.gazeBone(x)
        return [x1,x2,x3]
        
class multi_out_12(nn.Module):

    def __init__(self,):
        
        super(multi_out_12, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()

        self.eyeBone_right = layer.eyeBone_multiscale()
        self.mouthBone_right = layer.mouthBone_multiscale()

        self.eyeBone_left = layer.eyeBone_multiscale()
        self.mouthBone_left = layer.mouthBone_multiscale()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)

        stage9 = self.eyeBone_right(stage0)
        stage10 = self.mouthBone_right(stage0)

        stage11 = self.eyeBone_left(stage0)
        stage12 = self.mouthBone_left(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,
                stage9,stage10,stage11,stage12]

class multi_out_12_20210406(nn.Module):

    def __init__(self,):
        
        super(multi_out_12_20210406, self).__init__()

        self.baseBone = layer.backBone()
        
        self.eyeBone = layer.eyeBone_multiscale_20210406()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.recogBone = layer.RecognitionBone()

        self.eyeBone_right = layer.eyeBone_multiscale()
        self.mouthBone_right = layer.mouthBone_multiscale()

        self.eyeBone_left = layer.eyeBone_multiscale()
        self.mouthBone_left = layer.mouthBone_multiscale()

    def forward(self,x,idTargets=None):

        stage0 = self.baseBone(x)
        
        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.recogBone(stage0,idTargets)

        stage9 = self.eyeBone_right(stage0)
        stage10 = self.mouthBone_right(stage0)

        stage11 = self.eyeBone_left(stage0)
        stage12 = self.mouthBone_left(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,
                stage9,stage10,stage11,stage12]

class multi_out_10_20210526(nn.Module):

    def __init__(self,):

        super(multi_out_10_20210526, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.gazeBone = layer.GazeBone()
        self.angleRegBone = layer.AngleBoneReg()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.gazeBone(stage0)
        stage10 = self.angleRegBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9,stage10]

class pose(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.baseBone = layer.backBone()

        self.poseBone = layer.PoseBone()

    def forward(self,x):

        x = self.baseBone(x)
        yaw, pitch, roll = self.poseBone(x)

        return [yaw, pitch, roll]

class multi_out_9_20210615(nn.Module):

    def __init__(self,):

        super(multi_out_9_20210615, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.poseBone = layer.PoseBone()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.poseBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9]

class multi_out_11_20210721(nn.Module):

    def __init__(self,):

        super(multi_out_11_20210721, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.angleRegBone = layer.AngleBoneReg()
        self.poseBone = layer.PoseBone()
        self.FaceCls = layer.KLOccCls()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.angleRegBone(stage0)
        stage10 = self.poseBone(stage0)
        stage11 = self.FaceCls(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9,stage10,stage11]

class multi_out_13_20211008(nn.Module):

    def __init__(self,):

        super(multi_out_13_20211008, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.angleRegBone = layer.AngleBoneReg()
        self.poseBone = layer.PoseBone()
        self.FaceCls = layer.KLOccCls()
        self.genderBone = layer.genderClassify()
        self.ageBone = layer.ageClassify()


    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.angleRegBone(stage0)
        stage10 = self.poseBone(stage0)
        stage11 = self.FaceCls(stage0)
        stage12 = self.genderBone(stage0)
        stage13 = self.ageBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,
                stage7,stage8,stage9,stage10,stage11,stage12,stage13]

class multi_out_13_20210819(nn.Module):

    def __init__(self,):

        super(multi_out_13_20210819, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.binaryFaceBone = layer.binaryFaceBone()
        self.angleRegBone = layer.AngleBoneReg()
        self.poseBone = layer.PoseBone()
        self.FaceCls = layer.KLOccCls()

        self.eyeBone_right = layer.eyeBone_multiscale()
        self.eyeBone_left = layer.eyeBone_multiscale()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.binaryFaceBone(stage0)
        stage9 = self.angleRegBone(stage0)
        stage10 = self.poseBone(stage0)
        stage11 = self.FaceCls(stage0)

        stage12 = self.eyeBone_right(stage0)
        stage13 = self.eyeBone_left(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9,stage10,stage11,
                stage12,stage13]

class multi_out_8_20210803(nn.Module):

    def __init__(self,):

        super(multi_out_8_20210803, self).__init__()

        self.baseBone = layer.backBone()

        self.eyeBone = layer.eyeBone_multiscale()
        self.mouthBone = layer.mouthBone_multiscale()
        self.faceBone = layer.faceBone()
        self.detectBone = layer.detectBone_multiScale()
        self.emotionBone = layer.emotionBone_WITH_BG()
        self.angleBone = layer.angleBone_six()
        self.FaceAreaBone = layer.FaceAreaBone_seven_new()
        self.WrinkleBone = layer.WrinkleBone()

    def forward(self,x):

        stage0 = self.baseBone(x)

        stage1 = self.eyeBone(stage0)
        stage2 = self.mouthBone(stage0)
        stage3 = self.faceBone(stage0)
        stage4 = self.detectBone(stage0)
        stage5 = self.emotionBone(stage0)
        stage6 = self.angleBone(stage0)
        stage7 = self.FaceAreaBone(stage0)
        stage8 = self.WrinkleBone(stage0)

        return [stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8]