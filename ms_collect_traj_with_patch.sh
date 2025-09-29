nohup python collect_sweverified_trajectories.py \
    --exp_name qwen3coder_sweverify_20_patch \
    --args_path src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_glm45_with_patch.yaml \
> run_logs/qwen3coder_sweverify_20_patch.log 2>&1 &