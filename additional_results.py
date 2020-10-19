import numpy as np 
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns 

def get_results(basepath,cnn=True,merged=False):
    ### Loads results losses and accuracies files ###
    dirs = os.listdir(basepath)
    print(dirs)
    acclist = []
    losslist = []
    test_acclist = []
    dirs.sort()
    for i in range(len(dirs)):
        p = basepath + "/" + str(dirs[i]) + "/"
        acclist.append(np.load(p + "accs.npy"))
        losslist.append(np.load(p + "losses.npy"))
        test_acclist.append(np.load(p+"test_accs.npy"))
    print("enumerating through results")
    for i,(acc, l) in enumerate(zip(acclist, losslist)):
        print("acc: ", acc.shape)
        print("l: ", l.shape)
    else:
        return np.array(acclist), np.array(losslist),np.array(test_acclist)


def plot_results(pc_path, backprop_path,title,label1,label2,path3="",label3="",dataset="mnist",act_fn="tanh"):
    ### Plots initial results and accuracies ###
    acclist, losslist, test_acclist = get_results(pc_path)
    backprop_acclist, backprop_losslist, backprop_test_acclist = get_results(backprop_path)
    titles = ["Accuracies", "Losses", "Test Accuracies"]
    if path3 != "":
        p3_acclist, p3_losslist, p3_test_accslist = get_results(path3)
        p3_list = [p3_acclist,p3_losslist,p3_test_accslist]
    pc_list = [acclist, losslist, test_acclist]
    backprop_list = [backprop_acclist, backprop_losslist, backprop_test_acclist]
    print(acclist.shape)
    print(losslist.shape)
    print(test_acclist.shape)
    print(backprop_acclist.shape)
    print(backprop_losslist.shape)
    print(backprop_test_acclist.shape)
    for i,(pc, backprop) in enumerate(zip(pc_list, backprop_list)):
        xs = np.arange(0,len(pc[0,:]))
        mean_pc = np.mean(pc, axis=0)
        std_pc = np.std(pc,axis=0)
        mean_backprop = np.mean(backprop,axis=0)
        std_backprop = np.std(backprop,axis=0)
        print("mean_pc: ",mean_pc.shape)
        print("std_pc: ", std_pc.shape)
        fig,ax = plt.subplots(1,1)
        ax.fill_between(xs, mean_pc - std_pc, mean_pc+ std_pc, alpha=0.5,color='#228B22')
        plt.plot(mean_pc,label=label1,color='#228B22')
        ax.fill_between(xs, mean_backprop - std_backprop, mean_backprop+ std_backprop, alpha=0.5,color='#B22222')
        plt.plot(mean_backprop,label=label2,color='#B22222')
        if path3 != "":
            p3 = p3_list[i]
            mean_p3 = np.mean(p3, axis=0)
            std_p3 = np.std(p3,axis=0)
            ax.fill_between(xs, mean_p3 - std_p3, mean_p3+ std_p3, alpha=0.5,color='#228B22')
            plt.plot(mean_p3,label=label3)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(title + " " + str(titles[i]),fontsize=18)
        ax.tick_params(axis='both',which='major',labelsize=12)
        ax.tick_params(axis='both',which='minor',labelsize=10)
        if titles[i] in ["Accuracies", "Test Accuracies"]:
            plt.ylabel("Accuracy",fontsize=16)
        else:
            plt.ylabel("Loss")
        plt.xlabel("Iterations",fontsize=16)
        legend = plt.legend()
        legend.fontsize=14
        legend.style="oblique"
        frame  = legend.get_frame()
        frame.set_facecolor("1.0")
        frame.set_edgecolor("1.0")
        fig.tight_layout()
        #if titles[i] == "Test Accuracies":
        #    fig.savefig("./figures/"+str(dataset) + "_" + str(act_fn) + "_" +title +"_"+titles[i]+"_prelim_2.jpg")
        plt.show()


if __name__ == "__main__":
    dataset = "mnist"
    basepath = "relaxed_pc_experiments/further_experiments_"
    default_path = basepath + "default"
    backwards_weights_no_nonlinearities = basepath + "backwards_weights_no_nonlinearities"
    error_connections_backwards_weights = basepath + "error_alignment_backwards_weights"
    error_connections_no_nonlinearities = basepath + "error_connections_no_nonlinearities"
    combined_path = basepath + "full_construct"

    # pc vs learnt weights
    #plot_results(default_path,backwards_weights_no_nonlinearities,"Backwards Weights No Nonlinearities", "Standard Predictive Coding", "Backwards Weights No Nonlinearities",dataset = dataset)
    #pc vs feedback alignment
    #plot_results(default_path,error_connections_backwards_weights,"Error Connections Backwards Weights", "Standard Predictive Coding", "Error Connections Backwards Weights",dataset= dataset)
    #error alignment 
    #plot_results(default_path,error_connections_no_nonlinearities,"Error Connections No Nonlinearities", "Standard Predictive Coding", "Error Connnections No Nonlinearities",dataset=dataset)
    # Combined
    #plot_results(default_path, combined_path,"Combined Algorithm","Standard Predictive Coding", "Combined Relaxations",dataset=dataset)


    
    print("Relu")
    bpath = "relaxed_pc_experiments/layer_size_"
    act_fns = ["relu","tanh"]
    for act_fn in act_fns:
        basepath = bpath + str(act_fn) + "_"
        default_path = basepath + "default"
        learnt_error_connections = basepath + "error_connections"
        nonlinearities_path = basepath + "no_nonlinearities"
        combined_path = basepath + "full_construct"
        plot_results(default_path, nonlinearities_path,"Without Backwards Nonlinearity", "Standard Predictive Coding", "Without Nonlinear Derivative",dataset=dataset,act_fn="relu")#,path3=fa_path,label3="Feedback Alignment")
        plot_results(default_path, learnt_error_connections,"With Full Error Connectivity", "Standard Predictive Coding", "With Error Connections",dataset=dataset,act_fn="relu")
        plot_results(default_path, combined_path,"Combined Algorithm","Standard Predictive Coding", "Combined Relaxations",dataset=dataset,act_fn="relu")
    



