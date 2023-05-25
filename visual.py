import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import os
# import matplotlib

# matplotlib.use('Agg')

# this file will read in the sub_information and the prediction and ground truth data. Visualize the results including the step variance
# in each subject each trail. 

# set the parameters
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 3


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    alpha_1=0.8
    alpha_2=0.8
    l_width=1.5
    markersize = 8
    c_1='#7494FF'
    setp(bp['boxes'][0], color=c_1, linewidth=l_width, alpha = alpha_1)
    setp(bp['caps'][0], color=c_1,linewidth=l_width,alpha = alpha_1)
    setp(bp['caps'][1], color=c_1,linewidth=l_width,alpha = alpha_1)
    setp(bp['whiskers'][0], color=c_1,linewidth=l_width,alpha = alpha_1)
    setp(bp['whiskers'][1], color=c_1,linewidth=l_width,alpha = alpha_1)
    setp(bp['fliers'][0], alpha = alpha_1, markerfacecolor=c_1, marker='o',markeredgecolor=c_1)
    setp(bp['medians'][0], color=c_1,linewidth=l_width,alpha = alpha_1)

    c_2 = '#E88800'
    setp(bp['boxes'][1], color=c_2,linewidth=l_width,alpha = alpha_2)
    setp(bp['caps'][2], color=c_2,linewidth=l_width,alpha = alpha_2)
    setp(bp['caps'][3], color=c_2,linewidth=l_width,alpha = alpha_2)
    setp(bp['whiskers'][2], color=c_2,linewidth=l_width,alpha = alpha_2)
    setp(bp['whiskers'][3], color=c_2,linewidth=l_width,alpha = alpha_2)
    setp(bp['fliers'][1], alpha = alpha_1, markerfacecolor=c_2, marker='o',markeredgecolor=c_2)
    setp(bp['medians'][1], color=c_2,linewidth=l_width,alpha = alpha_2)

def iqr_fence(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    Lower_Fence = Q1 - (1.5 * IQR)
    Upper_Fence = Q3 + (1.5 * IQR)
    u = max(x[x<Upper_Fence])
    l = min(x[x>Lower_Fence])
    outlier_sum = ((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))).sum()
    return [u,l], outlier_sum

def plot_box_count(Normal,Shuffle,Stroke,sub_ID,G_stat):
    # first boxplot pair
    widths = 0.2
    alpha_1=0.5
    bp = boxplot(Normal[['predict','groundtruth']].values, positions=[0.5+0.5*sub_ID, 0.5+0.5*sub_ID+widths], widths=widths, patch_artist=True,notch=True)
    # get the summary statistics for the data
    setBoxColors(bp)
    Norman_pred_bound, outlier_sum = iqr_fence(Normal['predict'])
    G_stat.outlier_sum[0,0]+=outlier_sum
    Norman_grd_bound, outlier_sum = iqr_fence(Normal['groundtruth'])
    G_stat.outlier_sum[1,0]+=outlier_sum
    G_stat.stride_bound_pre_normal.append(Norman_pred_bound)
    G_stat.stride_bound_grd_normal.append(Norman_grd_bound)
    G_stat.stride_num[0,0] += Normal.shape[0]

    # # second boxplot pair
    bp = boxplot(Shuffle[['predict','groundtruth']].values, positions=[6+0.5*sub_ID, 6+0.5*sub_ID+widths], widths=widths, patch_artist=True,notch=True, )
    setBoxColors(bp)
    Shuffle_pred_bound, outlier_sum = iqr_fence(Shuffle['predict'])
    G_stat.outlier_sum[0,1]+=outlier_sum
    Shuffle_grd_bound,outlier_sum = iqr_fence(Shuffle['groundtruth'])
    G_stat.outlier_sum[1,1]+=outlier_sum
    G_stat.stride_bound_pre_Shuffle.append(Shuffle_pred_bound)
    G_stat.stride_bound_grd_Shuffle.append(Shuffle_grd_bound)
    G_stat.stride_num[0,1] += Shuffle.shape[0]

    # # thrid boxplot pair
    bp = boxplot(Stroke[['predict','groundtruth']].values, positions=[11.5+0.5*sub_ID, 11.5+0.5*sub_ID+widths], widths=widths, patch_artist=True,notch=True, )
    setBoxColors(bp)
    Stroke_pred_bound, outlier_sum = iqr_fence(Stroke['predict'])
    G_stat.outlier_sum[0,2]+=outlier_sum
    Stroke_grd_bound, outlier_sum = iqr_fence(Stroke['groundtruth'])
    G_stat.outlier_sum[1,2]+=outlier_sum
    G_stat.stride_bound_pre_Stroke.append(Stroke_pred_bound)
    G_stat.stride_bound_grd_Stroke.append(Stroke_grd_bound)
    G_stat.stride_num[0,2]+=Stroke.shape[0]

    ticks = ['Normal', 'Shuffle', 'Stroke']
    plt.xticks([2.75, 8.25, 13.75], ticks)

    # draw temporary red and blue lines and use them to create a legend
    c_1='#7494FF'
    c_2 = '#E88800'
    hB, = plt.plot([1, 1], c_1)
    hR, = plt.plot([1, 1], c_2)
    legend((hB, hR), ('Prediction values', 'Ground truth values'))
    hB.set_visible(False)
    hR.set_visible(False)

def compare_pre_ground(sub_info, test_results, ax):

    # plot the prediction values and ground truth values 
    start_steps = sub_info.groupby(['walkingtype'])['start_steps'].min()
    end_steps = sub_info.groupby(['walkingtype'])['end_steps'].max()

    plot_alpha = 0.2
    
    ax.plot(test_results, linewidth = 1.5)
    ax.axvspan(start_steps['Normal'], end_steps['Normal']+1, facecolor='#DBB40C', alpha = 0.4)
    ax.text((start_steps['Normal']+end_steps['Normal'])/2,np.min(test_results)*1.2, "Normal", ha='center')

    ax.axvspan(start_steps['Shuffle'], end_steps['Shuffle']+1, facecolor='#069AF3', alpha = plot_alpha)
    ax.text((start_steps['Shuffle']+end_steps['Shuffle'])/2,np.max(test_results)*.9, "Shuffle", ha='center')

    ax.axvspan(start_steps['Stroke'], end_steps['Stroke']+1, facecolor='#07DF62', alpha = plot_alpha)
    ax.text((start_steps['Stroke']+end_steps['Stroke'])/2,np.max(test_results)*.9, "Stroke", ha='center')
    ax.set_xlim(0, len(test_results))

def boxplot_all_test(all_test):  
    plt.figure()
    all_test=pd.melt(all_test, id_vars=['sub_ID','walkingtype'], value_vars=['predict', 'groundtruth'],
                     var_name='value type', value_name='stridelength')
    sns.boxplot(x='sub_ID', y='stridelength', hue='walkingtype', order=["predict", "groundtruth"],
                palette=sns.color_palette("husl", 3),
                # palette=["#3B49927F", "g", 'r'],
                data=all_test)
    plt.show()

def test_resutls_all(sub_info, test_results, param1,discard_option,subID):
    start_steps = sub_info.groupby(['walkingtype'])['start_steps'].min()
    end_steps = sub_info.groupby(['walkingtype'])['end_steps'].max()

    df_Normal=pd.DataFrame(test_results[start_steps['Normal']:end_steps['Normal'],:], columns=['predict','groundtruth'])
    df_Normal=df_Normal.assign(walkingtype='Normal')
    
    df_Shuffle=pd.DataFrame(test_results[start_steps['Shuffle']:end_steps['Shuffle'],:], columns=['predict','groundtruth'])
    df_Shuffle=df_Shuffle.assign(walkingtype='Shuffle')

    df_Stroke=pd.DataFrame(test_results[start_steps['Stroke']:end_steps['Stroke'],:], columns=['predict','groundtruth'])
    df_Stroke=df_Stroke.assign(walkingtype='Stroke')

    test_all_tyeps=pd.concat([df_Normal, df_Shuffle,df_Stroke], ignore_index=True, sort=False)
    test_all_tyeps=test_all_tyeps.assign(sub_ID=subID+1)

    return df_Normal, df_Shuffle, df_Stroke


def cal_RMSE_STD_all(test_results):
    all_RMSE = np.sqrt(np.mean((test_results[:,0] - test_results[:,1])**2))
    all_mean = np.mean(test_results,axis=0)
    all_std = np.std(test_results,axis=0)

    return all_RMSE, all_mean, all_std

def cal_RMSE_STD(sub_info, test_results, param1,discard_option):
    # calculate the RMSE for each perdiction task
    # calculate the RMSE for each walking type
    start_steps = sub_info.groupby(['walkingtype'])['start_steps'].min()
    end_steps = sub_info.groupby(['walkingtype'])['end_steps'].max()
    all_RMSE = np.sqrt(np.mean((test_results[:,0] - test_results[:,1])**2))
    
    normal_RMSE = np.sqrt(np.mean((test_results[start_steps['Normal']:end_steps['Normal'],0] 
                                   - test_results[start_steps['Normal']:end_steps['Normal'],1])**2))
    Shuffle_RMSE = np.sqrt(np.mean((test_results[start_steps['Shuffle']:end_steps['Shuffle'],0] 
                                   - test_results[start_steps['Shuffle']:end_steps['Shuffle'],1])**2))
    Stroke_RMSE = np.sqrt(np.mean((test_results[start_steps['Stroke']:end_steps['Stroke'],0] 
                                    - test_results[start_steps['Stroke']:end_steps['Stroke'],1])**2))
    df = {'subjectID': [sub_info['subjectID'].max()],'Normal': [normal_RMSE], 'Shuffle': [Shuffle_RMSE], 'Stroke':[Stroke_RMSE], 'All':[all_RMSE]}
    RMSE_df = pd.DataFrame(data=df)

    normal_mean =np.mean(test_results[start_steps['Normal']:end_steps['Normal'],:],axis=0)
    Shuffle_mean = np.mean(test_results[start_steps['Shuffle']:end_steps['Shuffle'],:],axis=0)
    Stroke_mean = np.mean(test_results[start_steps['Stroke']:end_steps['Stroke'],:],axis=0)
    normal_std = np.std(test_results[start_steps['Normal']:end_steps['Normal'],:],axis=0)
    Shuffle_std = np.std(test_results[start_steps['Shuffle']:end_steps['Shuffle'],:],axis=0)
    Stroke_std = np.std(test_results[start_steps['Stroke']:end_steps['Stroke'],:],axis=0)
    
    df = {'subjectID': [sub_info['subjectID'].max()],'Normal_mean_pre': [normal_mean[0]],'Normal_std_pre': [normal_std[0]], 
          'Shuffle_mean_pre': [Shuffle_mean[0]], 'Shuffle_std_pre': [Shuffle_std[0]],
          'Stroke_mean_pre':[Stroke_mean[0]],'Stroke_std_pre':[Stroke_std[0]],
          'Normal_mean_grd': [normal_mean[1]], 'Normal_std_grd': [normal_std[1]],
          'Shuffle_mean_grd': [Shuffle_mean[1]], 'Shuffle_std_grd': [Shuffle_std[1]],
          'Stroke_mean_grd':[Stroke_mean[1]], 'Stroke_std_grd':[Stroke_std[1]]}
    mean_std_df = pd.DataFrame(data=df)

    df=pd.DataFrame(test_results[start_steps['Normal']:end_steps['Normal'],:], columns=['predict','groundtruth'])
    df.to_csv(r'./outputs'+discard_option+param1+'all_test_normal.csv', mode='a', header=False, index=False)
    df=pd.DataFrame(test_results[start_steps['Shuffle']:end_steps['Shuffle'],:], columns=['predict','groundtruth'])
    df.to_csv(r'./outputs'+discard_option+param1+'all_test_Shuffle.csv', mode='a', header=False, index=False)
    df=pd.DataFrame(test_results[start_steps['Stroke']:end_steps['Stroke'],:], columns=['predict','groundtruth'])
    df.to_csv(r'./outputs'+discard_option+param1+'all_test_Stroke.csv', mode='a', header=False, index=False)

    return RMSE_df, mean_std_df

def cal_var(sub_info, test_results, sub_ID,param1,discard_option):
    # calculate the variance for each trial, each person
    # sub_info is the information for each trial and test_results are the prediction and truth values saved
    # return variance for each trial
    all_subject_var = sub_info.drop('start_steps', axis=1).drop('end_steps', axis=1)
    trail_var = np.zeros((len(sub_info.index),2))
    for index, (start_steps, end_steps) in enumerate(zip(sub_info['start_steps'].tolist(), sub_info['end_steps'].tolist())):
        steps_trail = test_results[start_steps:end_steps]
        # The variance for prediction values and ground truth 
        trail_var[index] = np.var(steps_trail, axis=0)   

    trail_var=pd.DataFrame(trail_var, columns=['var_pre','var_tru'])
    all_subject_var = pd.concat([all_subject_var, trail_var], axis=1)
    all_subject_var.to_csv(r'./outputs/'+discard_option+param1+sub_ID+'sub_var.csv',index=True)
    return all_subject_var

def plot_gride(data, col_wrap, plot_size):
    # Initialize a grid of plots with an Axes for each walk

    grid = sns.FacetGrid(data, col="subject ID", palette="tab20c", col_wrap=col_wrap, aspect=plot_size, 
                         height=5.5, sharex=False, sharey=False, despine=False)
    grid.set_titles(fontsize=20)
    grid.map_dataframe(sns.lineplot, x="index", y="variance", style="predic/truth", hue="walkingtype",
                       linewidth=2.5, markers=True, markersize=10, markevery=1)
    grid.fig.tight_layout(w_pad=0.4)
    grid.set_titles("")
    grid.set_xlabels("",fontsize=20, fontweight = 'bold')
    grid.set_ylabels("",fontsize=20, fontweight = 'bold')
    for ax in grid.axes.flatten():
        ax.tick_params(labelsize=20, which='major',width=1)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
    plt.show()

def compare_all_sub(var_subjects):
    sns.set_theme(style="ticks")
    # rename column name for better visualization
    var_subjects = var_subjects.rename({'subjectID': 'subject ID'}, axis=1) 
    var_subjects_1 = var_subjects.loc[(var_subjects['subject ID'] == 7) | (var_subjects['subject ID'] == 8)]
    var_subjects =  var_subjects.drop(var_subjects[var_subjects['subject ID']==7].index)
    var_subjects =  var_subjects.drop(var_subjects[var_subjects['subject ID']==8].index)
    plot_gride(data=var_subjects, col_wrap=4, plot_size=0.8)
    plot_gride(data=var_subjects_1, col_wrap=2, plot_size=1.6)

def delete_files(discard_option, param1):
    try:
        os.remove(r'./outputs'+discard_option+param1+'all_test_normal.csv')
        os.remove(r'./outputs'+discard_option+param1+'all_test_shuffle.csv')
        os.remove(r'./outputs'+discard_option+param1+'all_test_stroke.csv')
    except:
        print("no saved files found")


class Gait_statistics: 
    def __init__(self):
        # matrix to store the outlier number for each walking pattern 
        # pre_norm,pre_shuffle,pre_stroke
        # groundtruth_norm,groundtruth_shuffle,groundtruth_stroke
        self.outlier_sum=np.zeros((2,3))
        self.stride_num = np.zeros((1,3))
        self.stride_bound_pre_normal=[]
        self.stride_bound_grd_normal=[]
        self.stride_bound_pre_Shuffle=[]
        self.stride_bound_grd_Shuffle=[]
        self.stride_bound_pre_Stroke=[]
        self.stride_bound_grd_Stroke=[]


if __name__ == '__main__':

    num_sub = 10
    all_RMSE = pd.DataFrame({})
    all_mead_std = pd.DataFrame({})
    all_subjects_var = pd.DataFrame({})
    all_subjects_var = pd.DataFrame({})
    all_test = pd.DataFrame({})
    is_save = 1
    param1='/seq_length=1000/'
    G_stat = Gait_statistics()
    discard_option='/No_Discard/'   # '/Discard_first_last_step/' 

    if True:
        # delete files that is written with "a" mode.
        delete_files(discard_option,param1)
        fig, axs = plt.subplots(5, 2, figsize=(10, 20))
        for sub_ID, ax in zip(np.arange(num_sub),axs.flatten()):
            # ID in the dataset starts with 1
            sub_info = pd.read_csv(r'./outputs'+discard_option+param1+str(sub_ID+1)+'sub_info.csv', index_col=0)
            test_results=pd.read_csv(r'./outputs'+discard_option+param1+str(sub_ID+1)+'test.csv').values   

            # per_sub_test = test_resutls_all(sub_info,test_results, param1,discard_option,sub_ID)
            # all_test = pd.concat([all_test, per_sub_test], ignore_index=True, sort=False)

            per_sub_RMSE,mean_std_df = cal_RMSE_STD(sub_info, test_results, param1,discard_option)
            all_RMSE = pd.concat([all_RMSE, per_sub_RMSE], ignore_index=True, sort=False)
            all_mead_std = pd.concat([all_mead_std, mean_std_df], ignore_index=True, sort=False)

            var_subjects = cal_var(sub_info, test_results, str(sub_ID+1),param1,discard_option) 
    
            var_subjects_1 = pd.melt(var_subjects, id_vars=['subjectID','walkingtype','side'], value_vars=['var_pre', 'var_tru'], 
                                    var_name='predic/truth', value_name='variance', ignore_index=False)
            var_subjects_1 = var_subjects_1.reset_index()

            all_subjects_var = pd.concat([all_subjects_var, var_subjects_1], ignore_index=True, sort=False)

            compare_pre_ground(sub_info, test_results, ax) 

        plt.show()

        compare_all_sub(all_subjects_var)

        if is_save:
            all_RMSE.round(3).to_csv(r'./outputs'+discard_option+param1+'all_RMSE.csv',index=True)
            all_subjects_var.to_csv(r'./outputs'+discard_option+param1+'all_var.csv',index=True)
            all_mead_std.round(3).to_csv(r'./outputs'+discard_option+param1+'all_mean_std.csv',index=True)
        for sub_ID in np.arange(num_sub):
            # ID in the dataset starts with 1
            sub_info = pd.read_csv(r'./outputs'+discard_option+param1+str(sub_ID+1)+'sub_info.csv', index_col=0)
            test_results=pd.read_csv(r'./outputs'+discard_option+param1+str(sub_ID+1)+'test.csv').values   
            df_Normal, df_Shuffle, df_Stroke = test_resutls_all(sub_info,test_results,param1,discard_option,sub_ID)
            plot_box_count(df_Normal,df_Shuffle,df_Stroke,sub_ID,G_stat)
        plt.show()
        
        max_value_normal = max(max(sublist) for sublist in G_stat.stride_bound_grd_normal)

    test_normal = pd.read_csv(r'./outputs'+discard_option+param1+'all_test_normal.csv',names=['predict','groundtruth'],index_col=None).values
    test_shuffle = pd.read_csv(r'./outputs'+discard_option+param1+'all_test_shuffle.csv',names=['predict','groundtruth'],index_col=None).values
    test_stroke = pd.read_csv(r'./outputs'+discard_option+param1+'all_test_stroke.csv',names=['predict','groundtruth'],index_col=None).values
    all_type_results = np.concatenate((test_normal, test_shuffle, test_stroke), axis=0)

    all_normal_RMSE, all_normal_mean, all_normal_std = cal_RMSE_STD_all(test_normal)
    all_shuffle_RMSE, all_shuffle_mean, all_shuffle_std = cal_RMSE_STD_all(test_shuffle)
    all_stroke_RMSE, all_stroke_mean, all_stroke_std = cal_RMSE_STD_all(test_stroke)
    cal_RMSE_STD_all(all_type_results)
    print("data analysis and visualization finished")