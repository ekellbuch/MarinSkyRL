# Fully asynchronous DAPO dynamic sampling

## Decision

Fully asynchronous training supports `trainer.algorithm.dynamic_sampling.type=filter`. The training barrier discards
groups whose task outcomes have zero variance, requests fresh prompt groups, and waits until it has exactly
`policy_mini_batch_size` admissible groups. It never retries the rejected prompt and never trains a smaller batch.

The filter reads the reward source selected by `dynamic_sampling.informative_on` (`shaped` by default).
Select `unshaped` to filter on verifier outcomes before reward shaping. Length penalties and other shaping terms
can vary within an all-failure group and make it pass a filter configured to use shaped rewards.

`dynamic_sampling.type=replace` remains unsupported in fully asynchronous training. Replacement duplicates successful
groups inside a batch. It has different data-consumption and importance-weighting semantics from DAPO filtering.

## Motivation

DAPO removes prompt groups whose outcomes are all equal because they contribute no group-relative learning signal.
The synchronous trainer already implements this policy by collecting more generated batches. The fully asynchronous
trainer rejected every dynamic-sampling configuration before rollout workers started, so the same DAPO configuration
could not run with asynchronous generation.

The asynchronous trainer already has a barrier admission loop for stale, fully masked, incomplete, and otherwise
ineligible groups. It also preserves the invariant that every optimizer step receives a complete group batch. Dynamic
sampling belongs at this boundary, with a distinct disposition:

| Result | Prompt disposition | Counts toward DAPO sample budget |
| --- | --- | ---: |
| admissible, non-uniform outcomes | admit for training | yes |
| admissible, uniform outcomes | consume and request a fresh prompt | yes |
| stale, fully masked, incomplete, or missing required logprobs | retry the same prompt | no |

This separation prevents two incorrect behaviors. A stale attempt must not consume a new dataset example, while a
DAPO-rejected prompt must not loop through the same-source retry queue and produce the same uninformative group again.

The policy follows the DAPO paper and the refill behavior used by AReaL, verl, and OpenRLHF. These implementations
discard uniform groups and draw new prompts; they do not regenerate the rejected prompt as a failure retry. See the
[DAPO paper](https://arxiv.org/abs/2503.14476),
[AReaL workflow executor](https://github.com/areal-project/AReaL/blob/main/areal/infra/workflow_executor.py),
[verl asynchronous replay buffer](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/replay_buffer.py),
and [OpenRLHF sample generator](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/samples_generator.py).

## Barrier contract

The barrier evaluates completed groups in this order:

1. Apply algorithm-independent eligibility rules, including staleness, masking, physical group size, the configured
   group-advantage floor, and required behavior logprobs.
2. Route an ineligible group to same-prompt retry.
3. For an otherwise eligible group, read final outcomes from the configured reward source.
4. Consume groups whose population reward standard deviation does not exceed `min_reward_std`, and draw fresh prompts.
5. Admit groups that pass the filter and continue until the exact mini-batch size is available.

A group with only one final outcome remains informative, matching the existing filter behavior.

The barrier can hold accepted surplus groups for the next optimizer step. Existing buffer checkpointing remains the
owner of completed groups and same-prompt retries. Checkpoints occur at optimizer-step boundaries, so the DAPO sample
counter has no partially assembled step to restore.

## Sample budget and exhaustion

`max_sample_batches` bounds dataset cost per optimizer step. For fully asynchronous generation, one historical sample
batch is equivalent to `train_batch_size` otherwise eligible candidate groups. The maximum candidate count is therefore
`max_sample_batches * train_batch_size`. Accepted groups and uniform-outcome groups count against this limit. Attempts
rejected for staleness or invalid training data do not count because their prompts are retried.

If the barrier cannot assemble a complete batch within this budget, training raises an error. A zero or negative
`max_sample_batches` leaves the budget unlimited, matching synchronous behavior. Dataset exhaustion before a complete
batch is also an error; it does not weaken the batch invariant or silently reuse a DAPO-rejected prompt.

## Observability and validation

Admission metrics report DAPO filtering separately from eligibility failures. Each optimizer step records the number
of candidate groups inspected, uniform groups discarded, and accepted groups. Existing rejection metrics continue to
describe same-prompt retries.

CPU regression tests cover:

- replacement of a uniform group with a fresh prompt group;
- distinct routing for stale and uniform groups in the same admission sweep;
- filtering by unshaped outcomes when shaped rewards vary;
- exact mini-batch assembly with accepted surplus preserved;
- terminal sample-budget exhaustion without a short training batch; and
- rejection of `dynamic_sampling.type=replace` in fully asynchronous training.

The synchronous filter and the fully asynchronous filter must use the same outcome definition. Existing synchronous
dynamic-sampling tests remain part of the CPU gate.

## No-admission watchdog

For fully asynchronous filtering, `trainer.fully_async.dynamic_sampling_stall_timeout` defaults to 600 seconds.
The watchdog arms only after at least `policy_mini_batch_size` completed candidate groups have failed the
reward-spread filter since the last admission. It checks both arriving groups and an empty completion queue.
A newly admitted group resets the deadline and evidence count. With no completed filter evidence, the existing
generic admission and generation watchdogs retain their behavior.

When the armed deadline expires, the trainer raises `GenerationStalledError` with the machine-readable
`DYNAMIC_SAMPLING_NO_ADMISSIBLE_GROUPS` JSON diagnostic: candidate and admissible group counts, discarded
count/reasons, configured reward source, minimum/mean/maximum reward standard deviation, whether every observed
candidate is constant-zero or constant-one, and seconds since the last admission. Counts and reward summaries span
the current minibatch assembly. Check the scoring/parser outputs and recorded reward source before rerunning.
This operational bound does not change `min_reward_std`, group selection, or the required optimizer batch size.
