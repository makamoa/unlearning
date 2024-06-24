cd ..
python train.py --num_epochs=10 --batch_size=32 --data_dir='/media/makarem/Data/cifar100' --learning_rate=0.001 --log_dir='/media/makarem/Data/unlearn/logs/' --model='resnet18'
#python train.py --config_file='/media/makarem/Data/complex/logs/config.yaml'