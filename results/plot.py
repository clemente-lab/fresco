import matplotlib.pyplot as plt
import matplotlib

root = "C:\Users\Ryan Bernstein.RyanLaptop\Documents\work\\"

def plot_iters(filenames, legend, colors, title, axis):
    lines = []
    for filename in filenames:
        x = []
        y = []
        
        f = open(root+filename, 'r')
        for l in f:
            ls = l.split(" ")
            x.append(float(ls[0]))
            y.append(float(ls[1]))
            
        lines.append(x)
        lines.append(y)
        
    plt.figure()   
    #font = {'family' : 'normal',
     #   'weight' : 'bold',
      #  'size'   : 22}
    #matplotlib.rc('font', **font)
   # plt.title(title) 
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    for i in range(0, len(filenames)):
        plt.plot(lines[2*i], lines[2*i+1], colors[i])
    plt.axis(axis)
    plt.legend(legend, loc=4)
    plt.show()
           
def plot_threshs(filenames, legend, colors, title, axis):
    lines = []
    for filename in filenames:
        x = []
        y = []
        
        f = open(root+filename, 'r')
        for l in f:
            ls = l.split(" ")
            x.append(float(ls[0]))
            y.append(float(ls[1]))
            
        lines.append(x)
        lines.append(y)
        
    plt.figure()   
    #fsize = 10
    #font = {'family' : 'normal',
    #    'weight' : 'bold',
    #    'size'   : fsize}
    #matplotlib.rc('xtick', labelsize=fsize) 
    #matplotlib.rc('ytick', labelsize=fsize) 
    #matplotlib.rc('font', **font)
#    plt.title(title) 
    plt.xlabel("Starting Threshold")
    plt.ylabel("Accuracy")
    for i in range(0, len(filenames)):
        plt.plot(lines[2*i], lines[2*i+1], colors[i])
        
        
    plt.axis(axis)
    plt.legend(legend, loc=4)
    plt.show()
    
plot_iters(["iter_body_lr.txt", "iter_body_rf.txt", "iter_body_sv.txt"],
    ["Logistic Regression", "Random Forest", "Support Vector Machine"],
    ['r', 'b', 'g'],
    "Accuracy by Iterations, Study 449 Body Habitat",
    [0, 15, .7, .95])

plot_iters(["iter_host_lr.txt", "iter_host_rf.txt", "iter_host_sv.txt"],
    ["Logistic Regression", "Random Forest", "Support Vector Machine"],
    ['r', 'b', 'g'],
    "Accuracy by Iterations, Study 449 Host Individual",
    [0, 15, .5, .95])

plot_threshs(["thresh_body_lr_NORM.txt", "thresh_body_lr_LRND.txt", 
              "thresh_body_rf_NORM.txt", "thresh_body_rf_LRND.txt", 
              "thresh_body_sv_NORM.txt", "thresh_body_sv_LRND.txt"],
           ["Logistic Regression, Static", "Logistic Regression, Dynamic",
            "Random Forest, Static", "Random Forest, Dynamic",
            "Support Vector Machine, Static", "Support Vector Machine, Dynamic"],
           ['r', 'r--', 'b', 'b--', 'g', 'g--'],
           "Comparative Accuracy by Starting Threshold, Study 449 Body Habitat",
           [0.7, .99, .7, 1])
           
plot_threshs(["thresh_host_lr_NORM.txt", "thresh_host_lr_LRND.txt", 
              "thresh_host_rf_NORM.txt", "thresh_host_rf_LRND.txt", 
              "thresh_host_sv_NORM.txt", "thresh_host_sv_LRND.txt"],
           ["Logistic Regression, Static", "Logistic Regression, Dynamic",
            "Random Forest, Static", "Random Forest, Dynamic",
            "Support Vector Machine, Static", "Support Vector Machine, Dynamic"],
           ['r', 'r--', 'b', 'b--', 'g', 'g--'],
           "Comparative Accuracy by Starting Threshold, Study 449 Host Individual",
           [0.7, .99, .3, 1])
           