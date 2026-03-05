import json
from subprocess import call
from argparse import ArgumentParser
from glob import glob
import os
from omegaconf import OmegaConf
import subprocess
import stat


condor_template="""
# <<< Start of the input >>>>
# CPU / MEMORY
request_memory = <<N_RAM>>
request_cpus = <<N_CPUS>>
request_disk = <<N_DISK>>
# GPU
# request_gpus = <<N_GPUS>>
# requirements = (TARGET.CUDAGlobalMemoryMb > <<MIN_GPU_MEM>> ) && (TARGET.CUDAGlobalMemoryMb < <<MAX_GPU_MEM>>)
# job command
CmdLine = <<JOB_COMMAND>>
# <<<  end of input >>>>

#### start of the script
condor_scratch = $(_CONDOR_SCRATCH_DIR)
executable = /bin/bash
arguments = $(CmdLine)

# LOGS
### while change this path later
error  = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).err
output = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).out
log    = <<CONDOR_LOG_PATH>>/logs_condor/$(ClusterId).$(ProcId).log

# EXIT SETTINGS
on_exit_hold = (ExitCode =?= 3)
on_exit_hold_reason = "Checkpointed, will resume"
on_exit_hold_subcode = 2
periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 2) )

# Retry this job 5 times if non-zero exit code
# max_retries = 1

# max running price
+MaxRunningPrice = 5000
+RunningPriceExceededAction = "kill"
# add to job to queue
queue
"""


def main():
    
    parser = ArgumentParser()
    parser.add_argument('-p', "--path", default=None, type=str, help="Load the arguments from the config")
    parser.add_argument('-s', "--submit", default=False, action="store_true", help="Only submit if submit is True")
    parser.add_argument('-b', "--bid", default=100, type=str, help="Load the arguments from the config")
    
    args = parser.parse_args()
    job_config = OmegaConf.load(args.path)
    all_jobs_to_submit = job_config.ckpt_to_test
    print("Number of files to submit", len(all_jobs_to_submit) )

    condor_jobs_to_submit = []
    ### create the submit files
    for job_file in all_jobs_to_submit:

        condor_template_cp = condor_template
        if "debug" in job_file:
            print(f"\n*********************************************") 
            print(f"************ Skipping {job_file} **************")
            print(f"*********************************************\n") 
            continue

        print(f"\n************ Running {job_file} ************")

        # import pdb; pdb.set_trace()
        if "epoch" in job_file:

            method_ = job_config.CONDOR_JOB_CONFIG.args[2].split("/")[-1].split("_")[0]

            job_name = job_file.split("/")[-3] + "_testckpt_" + job_file.split("/")[-1].split(".")[0].replace("=","_")
            job_name = f"{method_}_{job_name}"
            
        elif "render_from_group_v2" in job_config.CONDOR_JOB_CONFIG.args[2] or "render_m2t_video" in job_config.CONDOR_JOB_CONFIG.args[2] :
            job_name = f"render_{job_file}"
        elif ".yaml" in job_file:
            job_name = job_file.split("/")[-1].split(".")[0]
        else:
            raise("Enter valid launch")

        condor_job_config = job_config.CONDOR_JOB_CONFIG

        ## prepare the condor sub file
        for k, v in condor_job_config.items():

            if k == "args":
                ### prepare command
                job_command = " ".join(list(condor_job_config.args))
                job_command = job_command.replace("<<EXP_CFG>>", job_file)
                condor_template_cp = condor_template_cp.replace("<<JOB_COMMAND>>", job_command)
                # print(f"<<JOB_COMMAND>>", job_command)
            elif k == "N_GPUS":
                if v == "set_from_device":
                    device = list(job_config.DEVICE)
                    n_gpus = len(device) if isinstance(device, list) else 1
                else:
                    n_gpus = v
                condor_template_cp = condor_template_cp.replace("<<N_GPUS>>", str(n_gpus))
                condor_template_cp = condor_template_cp.replace("# request_gpus", "request_gpus")
                condor_template_cp = condor_template_cp.replace("# requirements", "requirements")
                # request_gpus = <<N_GPUS>>
                # print(f"<<N_GPUS>>", n_gpus) 
            else:
                # print(f"<<{k}>>", v)
                condor_template_cp = condor_template_cp.replace(f"<<{k}>>", str(v))

        print(f"\nNew job config for :{job_name}")
        # print(condor_template_cp)

        ## out condor submit file
        condor_fname = os.path.join(condor_job_config.CONDOR_LOG_PATH, f"condor_job_files/{job_name}.sub")
        with open(condor_fname, 'w') as fp:
            fp.write(condor_template_cp)
        # os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
        print(f"Condor file created at: {condor_fname}")


        print(f"\n\nSubmitting jobs to condor...")
        os.system("bash /home/bthambiraja/projects/btraja-internship/external_repo/hands_only/_rsync/sync_to_cluster.sh")
        print("Called the following on the cluster: ")

        # cmd = f'cd {condor_job_config.PATH_TO_LAUNCH} && ' \
        #     f'echo $PWD > pwd.txt && ' \
        #     f'chmod +x {condor_fname} && ' \
        #     f'condor_submit_bid {args.bid} {condor_fname}'
        
        cmd = f'condor_submit_bid {args.bid} {condor_fname}'
        condor_jobs_to_submit.append(f"condor_submit_bid {args.bid} {condor_fname}")

        if args.submit:
            ssh_cmd = ["ssh",]
            ssh_cmd += ["bthambiraja@login.cluster.is.localnet"] + [cmd]
            subprocess.call(["ssh", "bthambiraja@login.cluster.is.localnet"] + [cmd])
    
    ## print jobs to submit
    print("\n\nJobs to submit:\n")
    print("\n".join(condor_jobs_to_submit))


if __name__ == "__main__":
    main()
