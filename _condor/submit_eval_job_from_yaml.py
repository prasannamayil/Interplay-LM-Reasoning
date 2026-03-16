import os
import subprocess
from argparse import ArgumentParser
from omegaconf import OmegaConf


condor_template = """
# <<< Start of the input >>>>
# CPU / MEMORY
request_memory = <<N_RAM>>
request_cpus = <<N_CPUS>>
request_disk = <<N_DISK>>
# GPU
request_gpus = <<N_GPUS>>
requirements = (TARGET.CUDAGlobalMemoryMb > <<MIN_GPU_MEM>> ) && (TARGET.CUDAGlobalMemoryMb < <<MAX_GPU_MEM>>)
# job command
CmdLine = <<JOB_COMMAND>>
# <<<  end of input >>>>

#### start of the script
# condor_scratch = $(_CONDOR_SCRATCH_DIR)
executable = /bin/bash
arguments = $(CmdLine)

# LOGS
error  = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).err
output = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).out
log    = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).log

# EXIT SETTINGS
on_exit_hold = (ExitCode =?= 3)
on_exit_hold_reason = "Checkpointed, will resume"
on_exit_hold_subcode = 2
periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 2) )

# max running price
+MaxRunningPrice = 5000
+RunningPriceExceededAction = "kill"
# add to job to queue
queue
"""


def build_context(run_name, ckpt, cfg):
    model1 = f"{cfg.MODEL_BASE_PATH}/{run_name}" if "MODEL_BASE_PATH" in cfg else run_name
    context = {
        "RUN_NAME": run_name,
        "model1": model1,
        "model_name": model1.split("/")[-1],  # shorthand: avoids <model1>.split("/")[-1] in templates
        "checkpoint": f"checkpoint-{ckpt}",
        "checkpoint_to_run": f"checkpoint-{ckpt}",
        "OUTPUT_PATH": cfg.OUTPUT_PATH,
    }
    # Include any other top-level scalar keys
    for k, v in cfg.items():
        if k not in context and isinstance(v, (str, int, float, bool)):
            context[k] = v
    return context


def resolve(arg, ctx):
    s = str(arg)
    for k, v in ctx.items():
        s = s.replace(f"<{k}>", repr(v))
    if any(tok in s for tok in ('.split(', '.replace(', '.strip(')):
        s = str(eval(s))
    else:
        s = s.replace("'", "").replace('"', "")
    return s


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str, help="Path to the eval yaml config")
    parser.add_argument('-s', '--submit', default=False, action='store_true', help="Submit jobs to condor")
    parser.add_argument('-b', '--bid', default=50, type=str, help="Condor bid")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.path)
    condor_job_config = cfg.CONDOR_JOB_CONFIG

    condor_job_files_dir = os.path.join(condor_job_config.CONDOR_LOG_PATH, "condor_job_files")
    os.makedirs(condor_job_files_dir, exist_ok=True)

    condor_jobs_to_submit = []

    for run_name in cfg.RUN_NAME:
        for ckpt in cfg.CHECKPOINT_TO_RUN:
            context = build_context(run_name, ckpt, cfg)

            for idx, model_cfg in enumerate(condor_job_config.ALL_MODELS_ARGS_TO_TEST):
                resolved_args = [resolve(arg, context) for arg in model_cfg.args]
                job_command = " ".join(resolved_args)
                job_name = f"{context['model_name']}_ckpt{ckpt}_m{idx}"

                t = condor_template
                for k, v in condor_job_config.items():
                    if k != "ALL_MODELS_ARGS_TO_TEST":
                        t = t.replace(f"<<{k}>>", str(v))
                t = t.replace("<<JOB_COMMAND>>", job_command)

                sub_file = os.path.join(condor_job_files_dir, f"{job_name}.sub")
                with open(sub_file, 'w') as fp:
                    fp.write(t)
                print(f"Created: {sub_file}")

                cmd = f"condor_submit_bid {args.bid} {sub_file}"
                condor_jobs_to_submit.append(cmd)

                if args.submit:
                    subprocess.call(cmd, shell=True)

    print("\n\nJobs to submit:\n")
    print("\n".join(condor_jobs_to_submit))


if __name__ == "__main__":
    main()
