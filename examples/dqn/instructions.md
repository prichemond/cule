DE-RAINBOW (bugged, in progress): python3 dqn_main.py --memory-capacity 500000 --num-ales 128 --t-max 100000 --rainbow --use-cuda-env --replay-frequency 1 --lr 1e-04 --multi-step 20 --learn-start 400 --max-grad-norm 10.0 --target-update 2000 
--priority-exponent 0.5 --max-episode-length 108000 --reward-clip --normalize

STANDARD RAINBOW:
python3 dqn_main.py --memory-capacity 500000 --num-ales 128 --t-max 100000 --rainbow --use-cuda-env
