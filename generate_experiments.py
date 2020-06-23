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
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " " + str(old_savepath)
    print(final_call)
    print(final_call, file=output_file)

condition="backwards_weights"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True"
    print(final_call)
    print(final_call, file=output_file)

condition="no_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)

condition="error_connections"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_connections True"
    print(final_call)
    print(final_call, file=output_file)

condition="full_construct"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    if old_savename != "None":
        old_savepath = save_path + "/" + str(old_savename) + "/" + str(s)
    else:
        old_savepath = "None"
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_error_connections True --use_backwards_weights True --use_backwards_nonlinearities False"
    print(final_call)
    print(final_call, file=output_file)