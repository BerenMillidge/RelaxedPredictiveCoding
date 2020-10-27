import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python main.py --neural_forward True"
output_file = open(generated_name, "w")
seeds = 5
act_fns = ["tanh","relu"]
for act_fn in act_fns:
    base_call = bcall + " --act_fn " + str(act_fn)
    cond = act_fn + "_"
    condition=cond + "default"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
        print(final_call)
        print(final_call, file=output_file)

    condition=cond + "backwards_weights"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)

    condition=cond + "no_nonlinearities"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False"
        print(final_call)
        print(final_call, file=output_file)

    condition=cond + "error_connections"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True"
        print(final_call)
        print(final_call, file=output_file)

    condition=cond + "full_construct"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
        print(final_call)
        print(final_call, file=output_file)
