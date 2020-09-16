import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python main.py"
output_file = open(generated_name, "w")
seeds = 5
activation_functions = ["relu", "tanh"]
for act_fn in activation_functions:
    condition = str(act_fn)
    base_call = bcall + " --activation_function " + str(act_fn)
    cond = condition + "_default"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
        print(final_call)
        print(final_call, file=output_file)

    cond = condition + "_backwards_weights"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)


    cond = condition + "_feedback_alignment"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights False"
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

    cond = condition + "_error_alignment"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --update_error_weights False"
        print(final_call)
        print(final_call, file=output_file)

    cond = condition + "_full_construct"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+cond + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+cond + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
        print(final_call)
        print(final_call, file=output_file)
