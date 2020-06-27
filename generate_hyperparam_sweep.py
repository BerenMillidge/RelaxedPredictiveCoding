import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 1
learning_rates = [0.0005,0.001,0.005,0.01,0.0001]
n_inference_steps = [10,50,100,200,500]
inference_learning_rates = [0.1,0.05,0.01]
for lr in learning_rates:
    for n_steps in n_inference_steps:
        for inference_lr in inference_learning_rates:
            condition=str(n_steps) + "_steps_"+str(inference_lr)+"inf_lr_"+str(lr)+"lr"
            for s in range(seeds):
                lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
                spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
                final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --learning_rate " + str(lr) + " --inference_learning_rate " + str(inference_lr) + " --n_inference_steps" + str(n_steps) + " --use_error_connections True"
                print(final_call)
                print(final_call, file=output_file)