## README

This folder contains the code of `SCINet` to train the models based on the generated data.

Getting Started:

1. After generating the generated data, (e.g. ETTh1), we can train the `SCINet` model with the following command (horizon=168)
    ```bash
    data_path='/yourpath/gen.npy'
    multiplier='1'
    CUDA_VISIBLE_DEVICES=0 python run_ETTh_gen.py --data ETTh1 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4 --batch_size 32 --dropout 0.5 --model_name etth1_M_I336_O168_lr5e-4_bs32_dp0.5_h4_s1l3 --gen_path $data_path --multiplier $multiplier --augment '' --syn_model_name 'aecgan-x1' 
    ```
    - `$data_path` means where you store your generated data.
    - `$multiplier` means how much generated data will be used for training the model. `1` means that it will use the same amount of generated data as the original training dataset. You can generate large amount of generated data for a better downstream performance.
    - Other hyper-parameters are inherent setting of the model, or you can tune these hyper-parameters for a better downstream performance.