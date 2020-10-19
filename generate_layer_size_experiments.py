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
l1_sizes = [100,200,300,500,1000]
l2_sizes = [20,50,100,200,500]
l3_sizes = [20,50,100,200,500]
activation_functions = ["relu", "tanh"]
for (l1_size, l2_size, l3_size) in zip(l1_sizes, l2_sizes, l3_sizes):
    bbcall = bcall + " --l1_size " + str(l1_size) + " --l2_size " + str(l2_size) + " --l3_size " + str(l3_size)
    for act_fn in activation_functions:
        condition = str(act_fn) + "_" + str(l1_size) + "_" + str(l2_size) + "_" + str(l3_size) + "_"
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