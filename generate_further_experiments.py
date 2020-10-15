import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 1
condition="default"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
    print(final_call)
    print(final_call, file=output_file)

condition="backwards_weights_no_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)


    print(final_call, file=output_file)

condition="error_connections_no_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)

condition="error_alignment_backwards_weights"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True"
    print(final_call)
    print(final_call, file=output_file)

condition="full_construct"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_weights True --use_backwards_weights True --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)