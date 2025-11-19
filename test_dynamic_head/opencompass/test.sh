export CUDA_VISIBLE_DEVICES=1,2,5
export PYTHONPATH="/home/ylong030/Retrieval_Head/test_dynamic_head:${PYTHONPATH}"

source ~/miniconda3/bin/activate opencompass
#python run.py eval/eval_dynamic_llada_head_ruler.py --dump-eval-details -r

#python run.py eval/eval_dynamic_llada_head_ruler.py --dump-eval-details -r

python run.py eval/eval_llada_diffusion_one_step_ruler.py --dump-eval-details -r
