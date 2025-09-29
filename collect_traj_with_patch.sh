nohup python collect_sweverified_trajectories.py \
    --exp_name claude_sweverify_50_patch \
    --args_path src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_claude4s_with_patch.yaml \
> run_logs/claude_sweverify_50_patch.log 2>&1 &