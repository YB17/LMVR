set -x

MODEL_PATH='./checkpoints/llava-7b-pretrain-no-mm-rep2/'
IMAGE_FOLDER='./test_dataset/'

python -m llava.eval.extract \
    --model-name $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --exp_name test \
    --conv-mode vicuna_v1_1
