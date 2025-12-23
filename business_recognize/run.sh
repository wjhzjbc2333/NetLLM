python run.py --adapt \
              --test \
              --frozen \
              --device cuda:0 \
              --plm-type llama \
              --plm-size large \
              --llm-dim 3072 \
              --lr 1e-4 \
              --num-epoch 300 \
              --save-checkpoint-per-epoch 100 \
              --seed 42 \
              --payload-max-len 256 \
              --test-file 'data/test.csv' \
              --train-file 'data/train.csv' \
              