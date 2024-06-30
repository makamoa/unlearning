cd ..
python train.py --num_epochs=100 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18'
python train.py --num_epochs=100 --batch_size=128 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.1 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18' --use_sam --rho=0.5
#python train.py --config_file='/media/makarem/Data/complex/logs/config.yaml'