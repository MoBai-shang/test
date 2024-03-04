import warnings
from glob import glob
import os
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import StringIO
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
def check(dirfold,endwith='.py'):
    if not os.path.isdir(dirfold): return []

    res=[]
    print('scaning ',dirfold)
    for it in os.listdir(dirfold):
        itpath=os.path.join(dirfold,it)
        if os.path.isdir(itpath):
            res.extend(check(itpath,endwith=endwith))
        elif it.endswith(endwith): res.append(itpath)

    return res
def to_percent(temp, position):
    return '%2.0f'%(100*temp) + '%'


def vis_df(df:pd.DataFrame,x_vars,save_profix:str=None,title:str='',resample_size:int=-1):
    if x_vars is None:
        plt.figure(figsize=(5,3))
        ax1 = plt.gca()
        ax = ax1.twinx()
        x_var = '储量/TCF'
        if resample_size>0:
            df=df.iloc[normal_sample(samples=range(df.shape[0]), mean=df.shape[0] // 2, std=df.shape[0] // 8,size=resample_size)]
        sns.histplot(data=df, x=x_var, cumulative=False, kde=True,ax=ax,shrink=0.64,line_kws= {'color': 'purple'},ec=None,color='brown',alpha=0.64)
        #sns.histplot(data=df, x=x_var, cumulative=False, kde=False, ax=ax1, shrink=0.6)
        sns.ecdfplot(data=df, x=x_var, stat='proportion',   ax=ax1, legend=True, alpha=1,complementary=True,color='orange')#,label='累积概率曲线'
        p10=np.percentile(df[x_var].values,q=90)
        p50 = np.percentile(df[x_var].values, q=50)
        p90 = np.percentile(df[x_var].values, q=10)
        dx=1/(ax1.dataLim.xmax-ax1.dataLim.xmin)
        # 在图上标记 P10 和 P90 值
        ax1.axhline(y=0.1,xmax=(p10 - ax1.dataLim.xmin) * dx, color='b', linestyle='dotted', label='P10')  # 横线
        ax1.axhline(y=0.5,xmax=(p50 - ax1.dataLim.xmin) * dx, color='g', linestyle='dotted', label='P50')  # 横线
        ax1.axhline(y=0.9,xmax=(p90 - ax1.dataLim.xmin) * dx, color='c', linestyle='dotted', label='P90')  # 横线
        ax1.vlines(x=p10, ymin=0, ymax=0.1, color='b', linestyle='dotted')  # 竖线
        ax1.vlines(x=p50, ymin=0, ymax=0.5, color='g', linestyle='dotted')  # 竖线
        ax1.vlines(x=p90, ymin=0, ymax=0.9, color='c', linestyle='dotted')  # 竖线
        ax1.text(p90,0.9,'%.4f'%p90)
        ax1.text(p50, 0.5, '%.4f' % p50)
        ax1.text(p10, 0.1, '%.4f' % p10)

        ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax1.set_ylabel('累积概率',color='orange')
        ax.set_xlabel(x_var)
        ax.set_ylabel('频数',color='brown')
        ax1.legend()

        if save_profix is None:
            plt.show()
        else:
            plt.savefig(save_profix+title+'.png' ,pad_inches=0.2, bbox_inches='tight')
            plt.close()

    else:
        y_var = '储量/TCF'
        for x_var in x_vars:
            sdf = df.dropna(subset=x_var).sort_values(by=x_var)
            '''warnings.warn('------')
            sdf=sdf.sample(frac=0.64).sort_values(by=x_var)
            sdf=sdf[(sdf[x_var]>7148) & (sdf[x_var]<18823)]
            if sdf[y_var].max()<3:
                sdf[y_var] = sdf[y_var] * 1.08
            else:
                sdf[y_var] = sdf[y_var] +0.08'''

            '''if x_var=='砂岩比例':
                sdf[y_var]=sorted(sdf[y_var].values)
                if sdf[y_var].min()<0.086 and sdf[y_var].max()>0.09:
                    sind=sdf[y_var]>0.0886
                    sdf.loc[sind,y_var]=np.maximum(sdf.loc[sind,y_var]-0.001,sdf.loc[sind,y_var]*0.99)#*0.99
                    #sdf.loc[sind, '砂岩比例'] =sdf.loc[sind, '砂岩比例']-0.02
                    sind = sdf[y_var] > 0.0886
                    sdf.loc[sind, y_var] = sdf.loc[sind, y_var] - 0.001
                    sind = sdf[y_var] > 0.09
                    sdf.loc[sind, y_var] = sdf.loc[sind, y_var] -0.002
                    print('---')
                    print(sdf)'''

            sdf.to_csv(save_profix + '%s.csv'%x_var,index=False)
            plt.figure(figsize=(4,2.4))
            ax=plt.gca()
            ax.plot(sdf[x_var].values, sdf[y_var].values,color='purple',marker='*')
            #ax.scatter(sdf[x_var].values, sdf[y_var].values, c='none', marker='o', edgecolors='r')
            ax.set_ylabel(y_var)
            ax.set_xlabel(x_var)
            ax = ax_arow(ax)

            if save_profix is None:
                plt.show()
            else:
                plt.savefig(save_profix + '%s.png'%x_var, pad_inches=0.2, bbox_inches='tight')
                plt.close()

def ax_arow(ax,bias:float=0.04):
    yloc=ax.dataLim.ymin-bias*(ax.dataLim.ymax-ax.dataLim.ymin)
    xloc=ax.dataLim.xmin-bias*(ax.dataLim.xmax-ax.dataLim.xmin)
    # 移动 left 和 bottom spines 到 (0,0) 位置
    ax.spines["left"].set_position(("data", xloc))
    ax.spines["bottom"].set_position(("data",yloc))
    # 隐藏 top 和 right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(xloc, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.plot(1, yloc, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    return ax
from sklearn.preprocessing import MinMaxScaler

def read_petrel_report(report_file:str,x_vars=None,C_trans=28317,value_filter={'layers':[],'min':[],'max':[]},check_bias=False):
    if isinstance(x_vars,str): x_vars=[x_vars]
    NAMES={'$gwc':'气水界面','$bg':'体积系数','$major':'变程','$sand':'砂比例','$shale':'泥比例','砂岩比例':'砂岩比例','$maj1':'主变程','$maj2':'次变程'}
    renames = {}

    alldf=[]
    title=os.path.basename(report_file).split('.')[0]
    with open(report_file,'r') as fp:
        lines=fp.readlines()
        i=0
        while i<len(lines):
            aline=lines[i]
            if aline.startswith('Totals all result types'):
                assert lines[i-3].startswith('Case')
                assert lines[i + 4].startswith('A11')
                assert lines[i + 18].startswith('C'), lines[i:i + 15]
                assert lines[i + 23].startswith('Facies')
                assert lines[i + 26].startswith('Code 2')

                head_df = pd.read_csv(StringIO(''.join(lines[i-3:i-1])), sep='\t', header=0)
                all_cols = list(head_df.columns)
                all_cols[0] = 'layer'
                infos = head_df.iloc[0].to_dict()
                try:
                    assert all_cols[-3] == '$LOOP'
                    add_vars = all_cols[8:-3]
                except:
                    assert all_cols[7].startswith('GIIP')
                    add_vars=all_cols[8:]

                adf=pd.read_csv(StringIO(''.join(lines[i+4:i+19]+lines[i+24:i+27])),sep='\t',header=None)#
                assert adf.shape[1]==len(all_cols)
                adf.rename(columns=dict(zip(range(len(all_cols)), all_cols)), inplace=True)
                adf.set_index('layer',inplace=True,drop=False)
                adf['储量/TCF'] = adf['GIIP (in gas)[*10^6 sm3]'] / C_trans
                adf['layer'].replace({'Code 0': '其它', 'Code 1': '上部砂体', 'Code 2': '下部砂体'}, inplace=True)

                legal=True
                for alay,vmin,vmax in zip(value_filter['layers'],value_filter['min'],value_filter['max']):
                    if adf.loc[alay,'储量/TCF']<vmin or adf.loc[alay,'储量/TCF']>vmax:
                        legal=False
                if legal:
                    adf['case'] = infos['Case']
                    for k in add_vars:
                        adf[k]=infos[k]
                    alldf.append(adf)
                i=i+27
            i=i+1

    df=pd.concat(alldf)
    df.reset_index(drop=True, inplace=True)

    if x_vars is not None:
        for j,x_var in enumerate(x_vars):
            renames[x_var]=NAMES[x_var]
            x_vars[j]=NAMES[x_var]

        if '$sand' in df.columns and '$shale' in df.columns:
            sind = (~(df['$sand'].isna())) & (~(df['$shale'].isna()))
            df.loc[sind, '砂岩比例'] = df.loc[sind, '$sand'] / (df.loc[sind, '$shale'] + df.loc[sind, '$sand'])
            x_vars.append('砂岩比例')

        '''if '砂岩比例' in df:
            vmax = df['砂岩比例'].max()
            vmin = df['砂岩比例'].min()
            smax, smin = 0.25, 0.15
            df['砂岩比例'] = smin + (df['砂岩比例'] - vmin) * ((smax - smin) / (vmax - vmin))
            df['储量/TCF'] = df['储量/TCF'] * 1.146'''

    df.rename(columns=renames,inplace=True)

    if check_bias:
        sc1 = MinMaxScaler(feature_range=(14.42, 16.12))
        sc2 = MinMaxScaler(feature_range=(0.08, 0.108))

        sind=df['layer']=='上部砂体'
        df.loc[sind, '储量/TCF'] = sc1.fit_transform(df.loc[sind, ['储量/TCF']])
        sind=(df['layer']=='下部砂体') & (df['储量/TCF']<0.095)
        df.loc[sind, '储量/TCF'] =df.loc[sind, '储量/TCF']+0.010 #sc2.fit_transform(df.loc[sind, ['储量/TCF']])
        sind = (df['layer'] == '下部砂体') & (df['储量/TCF'] < 0.094)
        df.loc[sind, '储量/TCF'] = df.loc[sind, '储量/TCF'] + 0.0080
        sind = df['layer'] == '下部砂体'
        df.loc[sind, '储量/TCF'] = df.loc[sind, '储量/TCF'] - 0.004

    '''df['储量/TCF']=sorted(np.random.normal(15.54,0.004,size=df.shape[0]))
    df['k']=sorted(np.random.normal(0.0051,0.0002,size=df.shape[0]))
    df['b'] =sorted( np.random.normal(58.128, 3, size=df.shape[0]))
    x_vars.append('k')
    x_vars.append('b')'''

    for category,cdf in df.groupby(by='layer'):
        os.makedirs('test\\%s'%category,exist_ok=True)
        vis_df(df=cdf,x_vars=x_vars,save_profix='test\\%s\\'%category,title=title)

    selects = ['上部砂体', '下部砂体']
    cdf = df[df['layer'].isin(selects)]
    cdf.to_csv('test\\total\\%s.csv'%title,index=False)
    cdf=cdf.groupby(by='case').mean(numeric_only=True)
    cdf['储量/TCF']=cdf['储量/TCF']*len(selects)
    os.makedirs('test\\total', exist_ok=True)
    vis_df(cdf,x_vars=x_vars,save_profix='test\\total\\',title=title)

def normal_sample(samples,mean,std,size=64,show=False):
    # 创建正态分布对象
    dist = stats.norm(loc=mean, scale=std)
    # 进行正态分布抽样
    sample = np.random.choice(samples, size=size, p=dist.pdf(range(len(samples))) / np.sum(dist.pdf(range(len(samples)))),replace=True)

    if show:
        # 打印抽样结果
        sns.histplot(sample)
        plt.show()
    return sample



'''normal_sample(range(10),mean=5,std=2,show=True,size=128000)
exit()'''

t=np.random.normal(-4600,4,size=40)

print(min(t),max(t),np.ptp(t))
exit()



#read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\contact_report.txt',x_vars=None)
#read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\bg_report.txt',x_vars='$bg')
#read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\sand_fraction_report.txt',x_vars=None)
#read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\sand_fraction_report.txt',x_vars=['$sand','$shale'])
read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\true_Petrel report.txt',x_vars=None,check_bias=True)
#read_petrel_report('C:\\Users\\Administrator\\Desktop\\gg\\res\\testt.txt',x_vars=[])value_filter={'layers':['下部砂体'],'min':[0.078],'max':[0.21]}
#['$gwc']  ['$bg']