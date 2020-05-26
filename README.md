# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Note:

Please do specify a folder to persist the checkpoints. Checkpoints are stored in a folder. For example, I use this one as a default: : ./checkpoints

# Note:

Please use a CUDA environment.


# Note: 

for train, we specify only the folder to persist the checkpoints. For eg: ./checkpoints

For predict, we specify the complete path including the checkpoint file. For eg: ./checkpoints/checkpoint_cl.vgg19.pth



# SAMPLE COMMAND LINE ARGUMENTS: for predict.py
_________________________________________________
python predict.py --checkpoint="./checkpoints/checkpoint_cl.vgg19.pth" --topk=3 'flowers/valid/16/image_06671.jpg'

python predict.py --checkpoint="./checkpoints/checkpoint_cl.densenet121.pth" --topk=3 'flowers/valid/16/image_06671.jpg'




# SAMPLE COMMAND LINE ARGUMENTS: for train.py
______________________________________________
python train.py --arch="vgg19" --epoch=3 --save_dir="./checkpoints" './flowers'

python train.py --arch="densenet121" --epoch=3 --save_dir="./checkpoints" './flowers'



