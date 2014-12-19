import matplotlib.pyplot as plt

def plot_crosstab(metric, title, width = 6,height = 4):
    ct = pd.crosstab(metric,df.rejected)
    rVals = ct[0].values
    aVals = ct[1].values
    labs = ct.index.values

    ind = np.arange(len(rVals))
    fig, ax = plt.subplots()
    barWidth = 0.35       # the width of the bars
    
    #tableau color palette
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.) 
        
    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False) 

    # Ensure that the axis ticks only show up on the bottom and left of the plot.  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", left="off", right="off", labelleft="on")
    
    ax.bar(ind,rVals,barWidth,color=tableau20[15],label="rejected",edgecolor='none')
    ax.bar(ind+barWidth,aVals,barWidth,color=tableau20[2],label="approved",edgecolor='none')
    ax.set_xticks(ind+barWidth)
    ax.set_xticklabels(labs)
    plt.title(title)
    plt.legend(loc='best')
    fig.set_size_inches(width,height)
    fig.savefig('../data/plots/'+title+'.png',dpi=100)

def genericBar(labs,vals, title, width = 6,height = 4):

    ind = np.arange(len(vals))
    fig, ax = plt.subplots()
    barWidth = 0.35       # the width of the bars
    
    #tableau color palette
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.) 
        
    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False) 

    # Ensure that the axis ticks only show up on the bottom and left of the plot.  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", left="off", right="off", labelleft="on")
    
    ax.bar(ind,vals,barWidth,color=tableau20[2],edgecolor='none')
    ax.set_xticks(ind+barWidth)
    ax.set_xticklabels(labs)
    plt.title(title)
    plt.legend(loc='best')
    fig.set_size_inches(width,height)
    fig.savefig('../data/plots/'+title+'.png',dpi=100)