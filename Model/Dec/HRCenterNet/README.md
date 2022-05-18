----test image------
python test.py --data_dir /path/to/images --log_dir /path/to/pretrained --output_dir /path/to/save/outputs
-------training-----
python train.py --train_csv_path data/train.csv --train_data_dir data/images \
                --val_csv_path data/val.csv --val_data_dir data/images/ --val \
                --batch_size 8 --epoch 80