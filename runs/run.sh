cd ..
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18'
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.001
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.005
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.01
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.02
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.05
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.1
python train.py --num_epochs=30 --batch_size=128 --data_dir='/home/gasanoe/HOME_SCRATCH_FOLDER/unlearning/media/cifar100' --learning_rate=0.1 --log_dir='/home/gasanoe/HOME_SCRATCH_FOLDER/unlearning/logs/' --model='resnet18' --use_sam --rho=0.5
# python train.py --num_epochs=30 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=1.0
#python train.py --config_file='/media/makarem/Data/complex/logs/config.yaml'