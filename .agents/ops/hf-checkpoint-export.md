# Exporting a training checkpoint to HF safetensors

Turns a banked megatron or fsdp2 checkpoint into a Hugging Face model directory. First proven
end-to-end 2026-07-28: `global_step_75` → 16 safetensors, 61.08 GB, `qwen3_moe`.

## There is no offline converter, and there should not be one

A megatron run writes a `torch.distributed.checkpoint` set (`__N_0.distcp` + `.metadata`) whose
tensors are Megatron-native: layer-stacked keys, grouped experts. Converting them to HF layout runs
through `bridge.save_hf_weights` (mbridge), which needs the Megatron runtime and a live process group
at the original parallel geometry. There is no laptop path.

`cloud/iris/export_hf_checkpoint.py` therefore does not reimplement the conversion. It re-runs the
trainer's own export by exploiting a branch the trainer already has: `FullyAsyncTrainer._train_loop`
checks whether the resumed step is at or past `max_steps`, and if so calls
`_handle_resume_at_max_steps`, which fires `on_train_end`, which makes the checkpoint callback
request an HF save. Setting `max_steps` to the checkpoint's own step produces a job that loads the
weights, exports them, and exits without training a step.

## Running one

```bash
cd /Users/benjaminfeuer/Documents/MarinSkyRL
export DC_AGENT_SECRET_ENV=/Users/benjaminfeuer/Documents/secrets.env
set -a; source "$DC_AGENT_SECRET_ENV"; set +a
export KUBECONFIG=~/.kube/coreweave-iris
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python

$PY -m cloud.iris.export_hf_checkpoint \
  --ckpt_path s3://marin-us-east-02a/iris/<job>/checkpoints \
  --step <N> \
  --rl_config <the config the run was TRAINED with> \
  --model_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --cluster cw-us-east-02a --num-nodes 4 --gpus-per-node 8 \
  --job-name export-<arm>-s<N>
```

Always `--dry-run` first and read back the resolved image, geometry and `export_path`.

- **`--rl_config` must be the config the run trained under**, not today's default. The resumed
  checkpoint has to match the parallel layout it was saved with.
- **Geometry must match the training geometry.** The configs pin placement per role
  (`policy_num_nodes` 2 + `ref_num_nodes` 2, 8 GPUs each), so 4x8 is not a preference — a different
  node/GPU split will not resolve the sharded load. `_validate_rl_config_topology` rejects an
  obviously wrong `--num-nodes` before launch, but it cannot catch every mismatch.
- Submit several at once and let iris queue them. Do not serialise by hand.
- Output lands at `<ckpt_path parent>/exports/global_step_<N>/policy/`.

The script sets `hf_hub_repo_id=null` and `enable_db_registration=false`: an export writes to durable
storage and publishes nothing. Publishing is a separate, owner-authorised step.

## Two failures worth not repeating

**Empty `train_data`.** The trainer builds its prompts dataset during construction, before it can
reach the resume-at-max-steps branch, and asserts the dataset is at least `train_batch_size`. The
sweep configs carry `data.train_data: []`, so an export that passes none dies with

```
AssertionError: dataset should be atleast as large as `train_batch_size` 64, got size 0
```

roughly eight minutes in, with the Ray head already up and four nodes allocated. The script now
always passes `--train_data` (default `DEFAULT_EXPORT_TRAIN_DATA`). The rows are never consumed —
zero steps run. Fixed in #184.

**Node-local `export_path`.** Left unset it derives to `<experiments_dir>/<job>/exports` on a worker,
while `HFHubUploadCallback` runs on the driver, so the callback finds nothing and the run ends with
an empty repo — after which the nodes are reclaimed and the export is gone. Every completed run in
the 2026-07 sweep lost its model this way. The launcher now defaults it to durable `s3://`; #162.

Also: the visible error in a failed export's `job summary` is usually a wall of `ray::IDLE` zombie
processes from teardown. That is noise. The real error sits hundreds of lines above it — read with
`--tail --max-lines 500`, not `--no-tail`, which caps at 1000 lines and hides the ending.

## Verifying an export actually produced a model

Check the object store, not the log. Reading it needs `https://cwobject.com` and the `iris-task-env`
credentials — see [iris-operator-scripts.md](iris-operator-scripts.md).

A complete 30B MoE export is **~61 GB across 16 shards plus 6 metadata files**. Three checks:

1. `model.safetensors.index.json` — `metadata.total_size` should agree with the summed file sizes,
   and `weight_map` should hold every tensor (18,867 for this model).
2. `config.json` — `model_type` `qwen3_moe`, `num_hidden_layers` 48, `hidden_size` 2048.
3. The shard count matches `model-000NN-of-000NN`.

A directory holding only a late shard means the export is still writing, not that it failed.

## Publishing

Publishing to `laion/` and registering in Supabase are standing defaults per ops once an export is
verified. They are separate from the export job, which pushes nothing.
