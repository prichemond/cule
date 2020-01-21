python3 vtrace_main.py --env-name BreakoutNoFrameskip-v4 --normalize --use-cuda-env --num-ales 4800 
--multiprocessing-distributed --t-max 10000000 --num-steps 20 --num-minibatches 300

(--num-minibatches doesn't have to be set now that dual training is disabled by default)
