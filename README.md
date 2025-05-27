# USenhance23_POAS
## USenhance23 Challenge: https://ultrasoundenhance2023.grand-challenge.org/
Ultrasound Image Enhancement Challenge 2023  
Team POAS (POSTECH, ASAN Medical Center)

## Dataset Preparation
```
trainroot`: Original Dataset train root  
|-- `breast`: Class  
    |-- `trainA`: Low Resolution Train Dataset   
    |-- `trainB`: High Resolution Train Dataset  
```
```
`testroot`: Original Dataset train root  
|-- `breast`: Class  
    |--`testA`: Low Resolution Train Dataset   
    |--`testB`: High Resolution Train Dataset  
```
## Train/Test Phase  

You can use `train.py` to start training

```
!CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
--dataroot /home/user9/us_train \
--testdataroot /home/user9/us_test \
--name test \
--n_epochs 200 \
--gpu_ids 1 \
--input_nc 1 \
--output_nc 1 \
--batch_size 4 \
--phase train \
--is_mtl \
--dataset_mode unaligned2
```

and use `test.py` to generate enhanced image

```
!CUDA_VISIBLE_DEVICES=0,1 \
python test.py \
--results_dir /home/user9/us_test/result \
--name test \
--model_suffix _A \
--dataset_mode unaligned2 \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--checkpoints_dir /home/user9/checkpoint \
--batch_size 24 \
--phase test \
--testdataroot /home/user9/us_test \
--dataroot /home/user9/us_test \
--serial_batches 1 \
--is_mtl 
```

**Model training parameters Link** https://pan.baidu.com/s/1VIsl7HpZ3DZfpUqAzzs8gA?pwd=kps3 **Extraction code:** kps3

