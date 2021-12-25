# 可复现性
reproductive = True
# 数据集路径
imgDirTrain = '/home/jiangtao/dataset/train/megaage_asian' 
imgTxtTrain = 'train.txt'
imgDirValid = '/home/jiangtao/dataset/train/megaage_asian'
imgTxtValid = 'Val.txt'

# 训练用到的数字
batchSize = 512 
workers = 32 
imgSize = 128 
numEpoches = 200
validInterval = 2 
savingEpoches = 10 

# 数据增强
cutmix_alpha = 1.0

# 模型
modelName = 'multi_out_13_20211008 '

# 损失
lossName = 'crossEntroy'

# 优化器参数设定
momentum = 0.9
weightdecay = 0.0005
scheduler = 'cosineAnnealing'

# 预训练模型
pretrained_type = 'pretrained'
pretrained_model = './models/multiScale_all_20200917.pkl' 

# 训练分支
trainingBranch = ['genderBone']

# model和log路径
log_path = None
saved_path = None