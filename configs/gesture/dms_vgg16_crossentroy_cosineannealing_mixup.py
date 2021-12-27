# 可复现性
reproductive = True
# 数据集路径
imgDirTrain = '/home/jiangtao/dataset/train/Gesture_static' 
imgTxtTrain = 'train.txt'
imgDirValid = '/home/jiangtao/dataset/train/Gesture_static'
imgTxtValid = 'val.txt'

# 训练用到的数字
batchSize = 512 
workers = 32 
imgSize = 128 
numEpoches = 200
validInterval = 2 
savingEpoches = 10 

# 数据增强
mixup_alpha = 1.0

# 模型
modelName = 'vgg16'
pretrained = True
numClasses = 3

# 损失
lossName = 'crossEntroy'

# 优化器参数设定
momentum = 0.9
weightdecay = 0.0005
scheduler = 'cosineAnnealing'

# model和log路径
log_path = None
saved_path = None