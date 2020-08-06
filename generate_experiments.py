import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 5
condition="default"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
    print(final_call)
    print(final_call, file=output_file)

condition="backwards_weights"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True"
    print(final_call)
    print(final_call, file=output_file)

condition="no_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)

condition="error_connections"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True"
    print(final_call)
    print(final_call, file=output_file)

condition="error_alignment"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --update_error_weights False"
    print(final_call)
    print(final_call, file=output_file)

condition="full_construct"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)

# activation function experiments.
condition="no_nonlinearities_tanh"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False --activation_function tanh"
    print(final_call)
    print(final_call, file=output_file)

condition="no_nonlinearities_sigmoid"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False --activation_function sigmoid"
    print(final_call)
    print(final_call, file=output_file)