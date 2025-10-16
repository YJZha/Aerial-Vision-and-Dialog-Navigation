#!/bin/bash
# ====== HAA-Transformer 单卡训练/评估脚本 (GPU:1) ======

# 使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 基础配置
ngpus=1
seed=0

OUT_DIR="../datasets/AVDN/et_haa"

flag="--root_dir ../datasets \
      --world_size ${ngpus} \
      --seed ${seed} \
      \
      --feedback student \
      --max_action_len 10 \
      --max_instr_len 100 \
      \
      --lr 1e-5 \
      --iters 200000 \
      --log_every 50 \
      --batch_size 4 \
      --optim adamW \
      \
      --ml_weight 0.2 \
      --feat_dropout 0.4 \
      --dropout 0.5 \
      \
      --nss_w 0.1 \
      --nss_r 0 \
      \
      --darknet_model_file ../datasets/AVDN/pretrain_weights/yolo_v3.cfg \
      --darknet_weight_file ../datasets/AVDN/pretrain_weights/best.pt \
      --eval_first True \
      \
      --pred_dir $OUT_DIR/preds"   # <<<<<<<<<< 新增参数，保存每个 sample 的结果

# ========== 训练 ==========
if [ "$1" = "train" ]; then
    echo ">>> 开始训练 HAA-Transformer (GPU:1)"
    python xview_et/main.py \
        --output_dir $OUT_DIR/train_run \
        $flag \
        2>&1 | tee $OUT_DIR/train.log

# ========== 评估 ==========
elif [ "$1" = "eval" ]; then
    echo ">>> 开始评估 HAA-Transformer (GPU:1)"
    python xview_et/main.py \
        --output_dir $OUT_DIR/eval_run \
        $flag \
        --resume_file $OUT_DIR/train_run/ckpts/best_val_unseen.pth \
        --inference True \
        --submit True \
        2>&1 | tee $OUT_DIR/eval.log

else
    echo "用法: bash $0 [train|eval]"
    exit 1
fi
