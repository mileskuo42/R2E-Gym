nohup python collect_sweverified_trajectories.py \
    --exp_name claude_sweverify_50 \
    --args_path src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_claude4s.yaml \
> run_logs/claude_sweverify_50.log 2>&1 &