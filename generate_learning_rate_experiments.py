import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python main.py"
output_file = open(generated_name, "w")
seeds = 2
dataset = "fashion"
bcall += " --dataset " + dataset
lrs = [0.01,0.005,0.001,0.0001]
activation_functions = ["relu", "tanh"]
for lr in lrs:
    bbcall = bcall + " --weight_clamp_val " + str(lr)
    for act_fn in activation_functions:
        condition = str(act_fn) + "_" + str(lr)
        base_call = bbcall + " --activation_function " + str(act_fn)
        cond = condition + "_default"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_no_nonlinearities"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False"
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_error_connections"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True"
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_full_construct"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
            print(final_call)
            print(final_call, file=output_file)

inference_lrs = [0.1,0.05,0.01,0.005]
activation_functions = ["relu", "tanh"]
for inference_lr in inference_lrs:
    bbcall = bcall + " --weight_clamp_val " + str(inference_lr)
    for act_fn in activation_functions:
        condition = str(act_fn) + "_" + "inference_lr_" + str(inference_lr)
        base_call = bbcall + " --activation_function " + str(act_fn)
        cond = condition + "_default"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_no_nonlinearities"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False"
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_error_connections"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True"
            print(final_call)
            print(final_call, file=output_file)

        cond = condition + "_full_construct"
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
            print(final_call)
            print(final_call, file=output_file)
