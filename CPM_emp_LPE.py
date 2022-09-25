###############################################################################
#                                                                             #
#   Connectome-based predictive modeling of empathy in adolescents            #
#       with and without the low-prosocial emotion specifier                  #
#                                                                             #
#   By: Drew E. Winters, PhD.                                                 #
#                                                                             #
###############################################################################



# python version 3.95
# Sys.setenv(RETICULATE_PYTHON = "C:\\Program Files\\Python39")



# Packages
  # data manipulation
import pandas as pd
import numpy as np
import glob, os
import re


  # machine learning 
from nilearn import connectome       # for connectivity measure
from sklearn.model_selection import permutation_test_score 
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.model_selection import GridSearchCV

  ## Stats
from scipy.stats import ttest_ind
rng = np.random.default_rng()

  ## plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

  ## system 
from joblib import parallel_backend ## for parallel processing 
import warnings
warnings.simplefilter("ignore")



          ### IMAGING DATA ####

# FILE NAMES LIST
import glob
csv_list = glob.glob(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\Subj_timeseries_denoised\ROI_*")

  # printing to ensure they are the right files
for i in range(0,len(csv_list)):
    print(os.path.basename(os.path.normpath(csv_list[i])))

  # test to make sure it is the right #
len(csv_list) == 86 # true



# EXTRACTING TIMESERIES

    # assigning each timeseries csv to a single list
      # we are only keeping the columns that are related to the nodes form the parcelization 
        # the others have confounds that were used in preprocessing 

  ## extracting time series of 164 nodes
time_s=[]
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12):
    for i in range(0,len(csv_list)):
        time_s.append(pd.read_csv(csv_list[i], 
            header= None).iloc[:,3:167].to_numpy())
    ## running test to make sure we have the right number of participants
len(time_s) == 86 # True
print("Total number of participants = ",len(time_s))

  ## checking the dimension of vars downloaded
for i in range(0,len(time_s)):
  print("Dimensions for participant #",i,"=",time_s[i].shape)


# EXTRACTIG HEAD MOTION
  ## here we extract the headmotiong variable to use as a covariate
head_m=[]
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
    for i in range(0,len(csv_list)):
        head_m.append(pd.read_csv(csv_list[i], header= None).iloc[:,169].to_numpy())
    ## test to ensure we have the right number  
len(head_m) == 86 # True

  # averaging headmotion for each participant
    ## will use this as a covariate later
head_m_ave = []
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
    for i in head_m:
        head_m_ave.append(np.average(i))
    ## test to ensure right #
len(head_m_ave) == 86 # True
    ## Assigning to dataframe
head_m_ave = pd.DataFrame({"h_move":head_m_ave})



# LABELS FOR ROIs

    ## here we extract whole brain ROI labels
      # and we reformat them to be more user friendly

  # Extracting labels 
labels = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\ROInames.csv", header=None).iloc[:,3:167]
labels = np.array(labels)[0]

  # modifying labels to shorter labels 
new_namess = []
for i in labels:
  new_namess.append(i.removeprefix('atlas.'))

new_namesss = []
for i in new_namess:
  new_namesss.append(i.removeprefix('networks.'))

new_names = new_namesss


  # Removing Suffix
import re

dll = []
for i in new_names:
  #print(i)
  dll.append(re.sub(r'\([L]\)','L', str(i)))

dlr = []
for i in dll:
  #print(i)
  dlr.append(re.sub(r'\([R]\)','R', str(i)))



dl = []
for i in dlr:
  #print(i)
  dl.append(re.sub(r'\([^()]*\)','', str(i)))



dls = []
for i in dl:
  #print(i)
  dls.append(re.sub(r' $','', str(i)))

dls2 = []
for i in dls:
  #print(i)
  dls2.append(re.sub(r' $','', str(i)))

dls3 = []
for i in dls2:
  dls3.append(re.sub(r' l',' L', str(i)))

dls4 = []
for i in dls3:
  dls4.append(re.sub(r' r',' R', str(i)))

new_names=np.array(dls4).astype('O')

  # test to ensrue we have the right number of labels 
len(new_names) == 164 # True




# COORDINATES FOR ROIs

coords = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\CONN_coordinates NO LABELS.csv", header=None)
coords.columns = ["x", "y", "z"]

  # test for correct number of nodes
len(coords) == 164 # True





            ##### BEHAVIORAL DATA AND DESCRIPTIVES ####

# READING IMAGING DATA
dda = pd.read_csv(r"D:\IU Box Sync\2 Dissertation & Qualifying exam\Rockland Data\Data_Rockland_SEM\2_4_19 newest data r code\2019_6_6_imaging_cases_86_FINAL.csv")
dda.shape
  # test to ensrue we dont have repeat IDs
np.where(dda.ID.value_counts()>1) ## none >1

# READING INDIVIDUAL ITEM CU TRAIT DATA 
ICUY=pd.read_csv(r"D:\IU Box Sync\2 Dissertation & Qualifying exam\Rockland Data\Data_Rockland_SEM\Assessments_rklnd_SEM\8100_ICUY_20180929.csv",header=1)
  # testing for duplicates
ICUY[['ID','VISIT']].value_counts()[np.where(ICUY[['ID','VISIT']].value_counts() >1)[0]] # participant A00065978 is duplicated 
dup = ICUY[['ID','VISIT']].value_counts()[np.where(ICUY[['ID','VISIT']].value_counts() >1)[0]].index[0][0]
ICUY = ICUY.drop(np.where(ICUY['ID'] == dup)[0][-1]) # droping the duplicate
  # formatting the visit labels to match
ICUY['VISIT'] = ICUY['VISIT'].replace(['V1'],'V2')
  # formating the column name
ICUY=ICUY.rename(columns={"VISIT": "visit"})

# MERGIG BRAIN AN CU DATA
df = dda.merge(ICUY, on=(['ID','visit']))
  # test to see if there are any duplicates
df.ID.value_counts()[np.where(df.ID.value_counts() >1)[0]] ## No duplicates
  # test to make sure we have the right number 
len(df) == 86 ## True
  # info on dataframe 
df.info()


# CALCULATING LPE SPECIFIER'S'
  ## reverse scoring 
df['DICUY_01']=(3-df['DICUY_01'])
df['DICUY_03']=(3-df['DICUY_03'])
df['DICUY_05']=(3-df['DICUY_05'])
df['DICUY_08']=(3-df['DICUY_08'])
df['DICUY_13']=(3-df['DICUY_13'])
df['DICUY_14']=(3-df['DICUY_14'])
df['DICUY_15']=(3-df['DICUY_15'])
df['DICUY_16']=(3-df['DICUY_16'])
df['DICUY_17']=(3-df['DICUY_17'])
df['DICUY_19']=(3-df['DICUY_19'])
df['DICUY_23']=(3-df['DICUY_23'])
df['DICUY_24']=(3-df['DICUY_24'])

  # dichotomizing for CU severity
df['DICUY_01_RC1'] = np.where(df['DICUY_01']>1,1,0)
df['DICUY_03_RC1'] = np.where(df['DICUY_03']>1,1,0)
df['DICUY_05_RC1'] = np.where(df['DICUY_05']>1,1,0)
df['DICUY_06_RC1'] = np.where(df['DICUY_06']>1,1,0)
df['DICUY_08_RC1'] = np.where(df['DICUY_08']>1,1,0)
df['DICUY_13_RC1'] = np.where(df['DICUY_13']>1,1,0)
df['DICUY_15_RC1'] = np.where(df['DICUY_15']>1,1,0)
df['DICUY_16_RC1'] = np.where(df['DICUY_16']>1,1,0)
df['DICUY_17_RC1'] = np.where(df['DICUY_17']>1,1,0)
df['DICUY_24_RC1'] = np.where(df['DICUY_24']>1,1,0)

  ## 4 Item method

df['sum_hi_cu4']=df[['DICUY_03_RC1','DICUY_05_RC1','DICUY_06_RC1','DICUY_08_RC1']].sum(axis=1)
df['hi_s_cu4'] = np.where(df['sum_hi_cu4']>=2,1,0)

df['hi_ex_cu4'] = np.where(df['sum_hi_cu4']>=3,1,0)

  ## 9 Item method
df['unconP_cu9_sum']= df[['DICUY_03_RC1','DICUY_15_RC1']].sum(axis=1)
df['lguilt_cu9_sum']= df[['DICUY_05_RC1','DICUY_13_RC1','DICUY_16_RC1']].sum(axis=1)
df['lemp_cu9_sum']= df[['DICUY_08_RC1','DICUY_17_RC1','DICUY_24_RC1']].sum(axis=1)

df['saff_s_cu9']= np.where(df['DICUY_01_RC1']>=2,1,0)
df['unconP_s_cu9']= np.where(df['unconP_cu9_sum']>=2,1,0)
df['lguilt_s_cu9']= np.where(df['lguilt_cu9_sum']>=2,1,0)
df['lemp_s_cu9']= np.where(df['lemp_cu9_sum']>=2,1,0)

df['sum_hi_s_cu9']=df[['saff_s_cu9','unconP_s_cu9','lguilt_s_cu9','lemp_s_cu9']].sum(axis=1)
df['hi_s_cu9'] = np.where(df['sum_hi_s_cu9']>=2,1,0)

df['saff_ex_cu9']= np.where(df['DICUY_01_RC1']>=3,1,0)
df['unconP_ex_cu9']= np.where(df['unconP_cu9_sum']>=3,1,0)
df['lguilt_ex_cu9']= np.where(df['lguilt_cu9_sum']>=3,1,0)
df['lemp_ex_cu9']= np.where(df['lemp_cu9_sum']>=3,1,0)

df['sum_hi_ex_cu9']=df[['saff_ex_cu9','unconP_ex_cu9','lguilt_ex_cu9','lemp_ex_cu9']].sum(axis=1)
df['hi_ex_cu9'] = np.where(df['sum_hi_ex_cu9']>=2,1,0)



# EXAMINING DESCRIPTIVES BY LPE SPECIFIER

  # 4-item
    ## split
np.sum(df.hi_s_cu4) #27
np.sum(df.hi_s_cu4)/len(df.hi_s_cu4) #31%
    ## extreme
np.sum(df.hi_ex_cu4) #12
np.sum(df.hi_ex_cu4)/len(df.hi_s_cu4) #13.9%

  # 9-item
    ## split
np.sum(df.hi_s_cu9) #24
np.sum(df.hi_s_cu9)/len(df.hi_s_cu9) #27.9%
    ## extreme
np.sum(df.hi_ex_cu9) #8
np.sum(df.hi_ex_cu9)/len(df.hi_s_cu9) #9.3%


# what I will do is create two groups to compare
  #> Those who qualify for any LPE = LPE
  #> THose who do not qualify for any LPE = normative

# summing all and seeing how many times each qualify
df["hi_all"] = (df.hi_ex_cu4 + df.hi_s_cu4 + df.hi_s_cu9 + df.hi_ex_cu9)
df["hi_all"].value_counts()
  #> those that qualify for LPE
  #> once = 10
  #> twice = 10
  #> thrice = 5
  #> all four times = 4

  # extreme and times qualifying
df["extreme"] =(df.hi_ex_cu4 + df.hi_ex_cu9)
df["extreme"].value_counts()
  #> once = 12
  #> twice = 4

  # split and times qualifying
df["split"] =(df.hi_s_cu4 + df.hi_s_cu9)
df["split"].value_counts()
  #> once = 22
  #> twice = 7

df["four"] =(df.hi_s_cu4 + df.hi_ex_cu4)
df["four"].value_counts()

df["nine"] =(df.hi_s_cu9 + df.hi_ex_cu9)
df["nine"].value_counts()

# looking at tables of the different coding methods 
df[["nine", "four"]].value_counts().unstack(fill_value=0)

df[["extreme", "split"]].value_counts().unstack(fill_value=0)

# examining numbers by each LPE
df[["hi_ex_cu9", "hi_s_cu9", "hi_ex_cu4", "hi_s_cu4"]].value_counts().unstack(fill_value=0)



# CREATING ONE COLUMN EITHER LPE (1) OR NO LPE (0)
df["hi_all_rc"] = df["hi_all"].apply(lambda x: 1 if x >= 1 else 0)
df["hi_all_rc"].value_counts()
  #> 57 "normative"
  #> 29 LPE
  ## test to ensure correct # of participants
len(df["hi_all_rc"]) == 86 # True

pd.set_option('display.max_rows', None)
# pd.Series(df.iloc[:,0:30].columns)
# pd.Series(df.columns)


  # agregating summary stats by 
pd.set_option('display.max_columns', None)
df[["hi_all_rc","PERSPECTIVE_TAKING", "EMPATHIC_CONCERN", "ICUY_TOTAL", "YSR_EXTERNALIZING_RAW"]].groupby("hi_all_rc").describe().loc[:,(slice(None),['mean','std'])].T

  # group by table by sex
df.groupby(["hi_all_rc","sex"]).size().unstack(fill_value=0)
  

# TESTING FOR GROUP DIFFERENCES
# X2 tests
from scipy.stats import chisquare
  # sex
pd.DataFrame(chisquare(df.groupby(["hi_all_rc","sex"]).size().unstack(fill_value=0), axis= 1), columns = ["sex: non-LPE","sex: LPE"], index= ["x2","p-val"]).T
  # race
pd.DataFrame(chisquare(df.groupby(["hi_all_rc","race"]).size().unstack(fill_value=0), axis= 1), columns = ["race: non-LPE","race: LPE"], index= ["x2","p-val"]).T


# T-tests
from scipy.stats import ttest_ind
rng = np.random.default_rng()

ttest_ind(df[df['hi_all_rc'] ==0].age,
          df[df['hi_all_rc'] ==1].age, 
          permutations=10000,
          random_state=rng
          )

ttest_ind(df[df['hi_all_rc'] ==0].tanner,
          df[df['hi_all_rc'] ==1].tanner, 
          permutations=10000,
          random_state=rng
          )

ttest_ind(df[df['hi_all_rc'] ==0].PERSPECTIVE_TAKING,
          df[df['hi_all_rc'] ==1].PERSPECTIVE_TAKING, 
          permutations=10000,
          random_state=rng
          )

ttest_ind(df[df['hi_all_rc'] ==0].EMPATHIC_CONCERN,
          df[df['hi_all_rc'] ==1].EMPATHIC_CONCERN, 
          permutations=10000,
          random_state=rng
          )

ttest_ind(df[df['hi_all_rc'] ==0].ICUY_TOTAL,
          df[df['hi_all_rc'] ==1].ICUY_TOTAL, 
          permutations=10000,
          random_state=rng
          )

ttest_ind(df[df['hi_all_rc'] ==0].YSR_EXTERNALIZING_RAW,
          df[df['hi_all_rc'] ==1].YSR_EXTERNALIZING_RAW, 
          permutations=10000,
          random_state=rng
          )



                                #### DEMOGRAPHICS ####
# CONTINUOUS VARS
pd.DataFrame(df[["age",
                "tanner",
                "PERSPECTIVE_TAKING",
                "EMPATHIC_CONCERN",
                "ICUY_TOTAL",
                "YSR_EXTERNALIZING_RAW"]].describe()).iloc[[1,2,3,7],:]


# SEX = MALE
pd.concat([pd.DataFrame({"male":df.sex.value_counts()}), pd.DataFrame({"%": df.sex.value_counts()/86})], axis = 1)

# RACE = WHITE
pd.concat([pd.DataFrame({"White": df.race.value_counts()}), pd.DataFrame({"%": df.race.value_counts()/86})], axis = 1)


# correlations
pd.DataFrame(df[["sex",
                "race",
                "age",
                "tanner",
                "PERSPECTIVE_TAKING",
                "EMPATHIC_CONCERN",
                "ICUY_TOTAL",
                "YSR_EXTERNALIZING_RAW"]]).corr()


    # p values
def calculate_pvalues(df):
    from scipy.stats import pearsonr
    import pandas as pd
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


calculate_pvalues(pd.DataFrame(df[["sex",
                "race",
                "age",
                "tanner",
                "PERSPECTIVE_TAKING",
                "EMPATHIC_CONCERN",
                "ICUY_TOTAL",
                "YSR_EXTERNALIZING_RAW"]]))


## By LPE demographics
  # agregating summary stats by 
pd.set_option('display.max_columns', None)
df[["hi_all_rc",
          "age",
          "tanner",
          "PERSPECTIVE_TAKING",
          "EMPATHIC_CONCERN",
          "ICUY_TOTAL",
          "YSR_EXTERNALIZING_RAW"]].groupby("hi_all_rc").describe().loc[:,(slice(None),['mean','std'])].T

  # group by table by sex
df.groupby(["hi_all_rc","sex"]).size().unstack(fill_value=0)
df.groupby(["hi_all_rc","sex"]).size().unstack(fill_value=0)/86
  # group table by race
df.groupby(["hi_all_rc","race"]).size().unstack(fill_value=0)
df.groupby(["hi_all_rc","race"]).size().unstack(fill_value=0)/86
  




              #### EXTRACTING EMPATHY BY LPE ####

# SUBSETTING DATAFRAMES BY LPE
  # high
df_hi = df[df['hi_all_rc'] ==1]
# df_hi.shape
  # testing
len(df_hi) == len(np.where(df["hi_all_rc"] == 1)[0]) # True
  # low or normative
df_lo = df[df['hi_all_rc'] ==0]
# df_lo.shape
  # testing 
len(df_lo) == len(np.where(df["hi_all_rc"] == 0)[0]) # True

  # extracting empathy by LPE
  ## LPE
c_emp_hi = df_hi["PERSPECTIVE_TAKING"]
a_emp_hi = df_hi["EMPATHIC_CONCERN"]
# emp_hi = (c_emp_hi + a_emp_hi)
# c_emp_hi.shape
  ## No LPE (normative)
c_emp_lo = df_lo["PERSPECTIVE_TAKING"]
a_emp_lo = df_lo["EMPATHIC_CONCERN"]
# emp_lo = (c_emp_lo + a_emp_lo)
# c_emp_lo.shape




                                #### CONNECTIVITY MATRIX ####
# CALCULATING CONNECTOME
from nilearn import connectome
connectivity = connectome.ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True) 

fc_mat = connectivity.fit_transform(time_s)
  # testing 
len(fc_mat) == 86 # True


# creating an indicies matrix to help IDing edges later
tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = list(range(0,fc_mat[0].shape[0]))
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\upper_index.csv")




                        #### IDENTIFYING POSITIVE AND NEGATIVE NODES ####

## FUNCTION USED TO ID POSITIVE AND NEGATIVE REGIONS
def train_cpm(ipmat, pheno):
    """
    Accepts input matrices and pheno data
    Returns model
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    cpm: in cpm we select the most significant edges for subjects. so each subject
         have a pair set of edges with positive and negative correlation with behavioral subjects.
         It's important to keep both set in final regression task.  
    posedges: positive edges are a set of edges have positive
              correlatin with behavioral measures
    negedges: negative edges are a set of edges have negative
              correlation with behavioral measures
    """
    from scipy import stats
    import numpy as np

    cc=[stats.pearsonr(pheno,im) for im in ipmat]
    rmat=np.array([c[0] for c in cc])
    pmat=np.array([c[1] for c in cc])
    posedges=(rmat > 0) & (pmat < 0.005)
    posedges=posedges.astype(int)
    negedges=(rmat < 0) & (pmat < 0.005)
    negedges=negedges.astype(int)
    pe=ipmat[posedges.flatten().astype(bool),:]
    ne=ipmat[negedges.flatten().astype(bool),:]
    pe=pe.sum(axis=0)/2
    ne=ne.sum(axis=0)/2

    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)
    else:
        fit_neg=[]

    return fit_pos,fit_neg,posedges,negedges

      ### NOTE we tried multiple thresholds and found that 0.005 was the best fit (hyper parameter tuning)




# EXTRACTING LPE - COGNITIVE
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 1].T, c_emp_hi)
  ## now I have positive and negative indicies to pull from for my selection

np.sum(posedges)
np.sum(negedges)

  
  # creating matrix IDing identified regions to be use for later
tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = posedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\c_emp_hi_pos.csv")

# finding edge indicies
c_hi_pos_indicies = np.where(tri_df == 1)
  # listing indicies 
# pd.DataFrame(c_hi_pos_indicies).T
  # using indicies to identify edge names
c_hi_pos_edge_names = new_names[c_hi_pos_indicies[0]] + "__" + new_names[c_hi_pos_indicies[1]]
# pd.Series(c_hi_pos_edge_names)



tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = negedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\c_emp_hi_neg.csv")

  # finding edge indicies
c_hi_neg_indicies = np.where(tri_df == 1)
    # listing indicies 
# pd.DataFrame(c_hi_neg_indicies).T
    # using indicies to identify edge names
c_hi_neg_edge_names = new_names[c_hi_neg_indicies[0]] + "__" + new_names[c_hi_neg_indicies[1]]
# pd.Series(c_hi_neg_edge_names)


  # EXTRACTING POSITIVE AND NEGATIVE EDGES
c_hi_pos_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] ==1].shape[0]):
  c_hi_pos_edge.append(fc_mat[df['hi_all_rc'] ==1][i][posedges==1])
c_hi_pos_edge = np.array(c_hi_pos_edge)

c_hi_neg_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] ==1].shape[0]):
  c_hi_neg_edge.append(fc_mat[df['hi_all_rc'] ==1][i][negedges==1])
c_hi_neg_edge = np.array(c_hi_neg_edge)





# EXTRACT HIGH - AFFECTIVE
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 1].T, a_emp_hi)
## now I have positive and negative coeffs to pull from for my selection 

np.sum(posedges)
np.sum(negedges)

  # creating matrix IDing identified regions to be use for later
tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = posedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\a_emp_hi_pos.csv")

# finding edge indicies
a_hi_pos_indicies = np.where(tri_df == 1)
  # listing indicies 
# pd.DataFrame(a_hi_pos_indicies).T
  # using indicies to identify edge names
a_hi_pos_edge_names = new_names[a_hi_pos_indicies[0]] + "__" + new_names[a_hi_pos_indicies[1]]
# pd.Series(a_hi_pos_edge_names)



tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = negedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\a_emp_hi_neg.csv")

# finding edge indicies
a_hi_neg_indicies = np.where(tri_df == 1)
  # listing indicies 
# pd.DataFrame(a_hi_neg_indicies).T
  # using indicies to identify edge names
a_hi_neg_edge_names = new_names[a_hi_neg_indicies[0]] + "__" + new_names[a_hi_neg_indicies[1]]
# pd.Series(a_hi_neg_edge_names)


  # EXTRACTING POSITIVE AND NEGATIVE EDGES
a_hi_pos_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] ==1].shape[0]):
  a_hi_pos_edge.append(fc_mat[df['hi_all_rc'] ==1][i][posedges==1])
a_hi_pos_edge = np.array(a_hi_pos_edge)

a_hi_neg_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] ==1].shape[0]):
  a_hi_neg_edge.append(fc_mat[df['hi_all_rc'] ==1][i][negedges==1])
a_hi_neg_edge = np.array(a_hi_neg_edge)







# EXTRACTING NO LPE - COGNITIVE 
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 0].T, c_emp_lo)
## now I have positive and negative coeffs to pull from for my selection 

np.sum(posedges)
np.sum(negedges)

  # creating matrix IDing identified regions to be use for later
tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = posedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\c_emp_lo_pos.csv")

# finding edge indicies
c_lo_pos_indicies = np.where(tri_df == 1)
  # listing indicies 
# pd.DataFrame(c_lo_pos_indicies).T
  # using indicies to identify edge names
c_lo_pos_edge_names = new_names[c_lo_pos_indicies[0]] + "__" + new_names[c_lo_pos_indicies[1]]
# pd.Series(c_lo_pos_edge_names)




tri = np.zeros((164, 164))
tri[np.triu_indices(164, 1)] = negedges
tri_df = pd.DataFrame(tri)
tri_df.columns = new_names
tri_df.index = new_names
# tri_df.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\c_emp_lo_neg.csv")

# finding edge indicies
c_lo_neg_indicies = np.where(tri_df == 1)
  # listing indicies 
# pd.DataFrame(c_lo_neg_indicies).T
  # using indicies to identify edge names
c_lo_neg_edge_names = new_names[c_lo_neg_indicies[0]] + "__" + new_names[c_lo_neg_indicies[1]]
# pd.Series(c_lo_neg_edge_names)



  # EXTRACTING POSITIVE AND NEGATIVE EDGES
c_lo_pos_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] == 0].shape[0]):
  c_lo_pos_edge.append(fc_mat[df['hi_all_rc'] == 0][i][posedges == 1])
c_lo_pos_edge = np.array(c_lo_pos_edge)

c_lo_neg_edge = []
for i in range(0, fc_mat[df['hi_all_rc'] == 0].shape[0]):
  c_lo_neg_edge.append(fc_mat[df['hi_all_rc'] == 0][i][negedges == 1])
c_lo_neg_edge = np.array(c_lo_neg_edge)






                        #### ML models ####

# ML MODEL
  # packages
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.model_selection import GridSearchCV

  # setting up ML model 
    #> logic
      #> we are creating a model with elastic net in order to reduce over fitting 
      #> we are hyperparameter tuning for both alpha and L1
      #> We are doing a nested cross validation
        #> hyperparemeter tuning iwht 3 folds
        #> CV with 5 folds
model = make_pipeline(ElasticNet(random_state=42)) # StandardScaler()
  # parameters for hyperparamter tuning 
param_grid = {"elasticnet__alpha": np.logspace(-2, 0, num=20),'elasticnet__l1_ratio': np.logspace(-1.5, 0, num=20)}
  # setting up nested cross validation 
from sklearn.model_selection import StratifiedKFold
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  # nested CV model
model_s = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=inner_cv)




                #### ML MODEL: AFF EMP - HIGH CU ####

## AFFECTIVE HIGH POSITIVE MODEL__________________

  # adding controls   ## pd.DataFrame(a_hi_pos_edge).mean(axis=1)
a_hi_pos_edge_c = pd.concat([pd.DataFrame(a_hi_pos_edge),pd.concat([df_hi[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==1]], axis=1).set_index(pd.DataFrame(a_hi_neg_edge).index)],axis=1)
  # ML model

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  aff_hi_pos_res = cross_validate(model_s, a_hi_pos_edge_c, a_emp_hi, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-aff_hi_pos_res['train_neg_mean_squared_error'].mean():.3f} +/- {aff_hi_pos_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{aff_hi_pos_res['train_r2'].mean():.3f} +/- {aff_hi_pos_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-aff_hi_pos_res['train_neg_mean_absolute_error'].mean():.3f} +/- {aff_hi_pos_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-aff_hi_pos_res['test_neg_mean_squared_error'].mean():.3f} +/- {aff_hi_pos_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{aff_hi_pos_res['test_r2'].mean():.3f} +/- {aff_hi_pos_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-aff_hi_pos_res['test_neg_mean_absolute_error'].mean():.3f} +/- {aff_hi_pos_res['test_neg_mean_absolute_error'].std():.3f}")



  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(aff_hi_pos_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, a_hi_pos_edge_c, a_emp_hi, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


aff_hi_pos_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
aff_hi_pos_errors


    ## DECISION - LPE POSITIVE AFFECTIVE MODEL 
      #> generalizes adn better than null model - good




  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(a_hi_pos_edge_c, a_emp_hi)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    a_hi_pos_edge_c, 
    a_emp_hi, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 ADN MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="blue")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-4,1.25)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_aff_pos_LPE.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()



  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(a_hi_pos_edge_c, a_emp_hi,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="blue", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Positive Connectivity in Low Prosocial Emotion Specifier')
plt.text(18,8,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_aff_pos_LPE.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()






## AFFECTIVE HIGH NEGATIVE MODEL_________________________

  # adding controls
a_hi_neg_edge_c = pd.concat([pd.DataFrame(a_hi_neg_edge),pd.concat([df_hi[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==1]], axis=1).set_index(pd.DataFrame(a_hi_neg_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  aff_hi_neg_res = cross_validate(model_s, a_hi_neg_edge_c, a_emp_hi, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-aff_hi_neg_res['train_neg_mean_squared_error'].mean():.3f} +/- {aff_hi_neg_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{aff_hi_neg_res['train_r2'].mean():.3f} +/- {aff_hi_neg_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-aff_hi_neg_res['train_neg_mean_absolute_error'].mean():.3f} +/- {aff_hi_neg_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-aff_hi_neg_res['test_neg_mean_squared_error'].mean():.3f} +/- {aff_hi_neg_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{aff_hi_neg_res['test_r2'].mean():.3f} +/- {aff_hi_neg_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-aff_hi_neg_res['test_neg_mean_absolute_error'].mean():.3f} +/- {aff_hi_neg_res['test_neg_mean_absolute_error'].std():.3f}")


  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(aff_hi_neg_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, a_hi_neg_edge_c, a_emp_hi, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


aff_hi_neg_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
aff_hi_neg_errors



  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(a_hi_neg_edge_c, a_emp_hi)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    a_hi_neg_edge_c, 
    a_emp_hi, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 ADN MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="green")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-3,1.25)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_aff_neg_LPE.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()




  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(a_hi_neg_edge_c, a_emp_hi,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="green", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Negative Connectivity in Low Prosocial Emotion Specifier')
plt.text(18,8,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_aff_neg_LPE.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()



              #### ML MODEL: COG EMP - HIGH CU ####

## COGNITIVE HIGH POSITIVE MODEL_______________

  # adding controls
c_hi_pos_edge_c = pd.concat([pd.DataFrame(c_hi_pos_edge),pd.concat([df_hi[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==1]], axis=1).set_index(pd.DataFrame(a_hi_neg_edge).index)],axis=1)


from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  cog_hi_pos_res = cross_validate(model_s, c_hi_pos_edge_c, c_emp_hi, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-cog_hi_pos_res['train_neg_mean_squared_error'].mean():.3f} +/- {cog_hi_pos_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{cog_hi_pos_res['train_r2'].mean():.3f} +/- {cog_hi_pos_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-cog_hi_pos_res['train_neg_mean_absolute_error'].mean():.3f} +/- {cog_hi_pos_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-cog_hi_pos_res['test_neg_mean_squared_error'].mean():.3f} +/- {cog_hi_pos_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{cog_hi_pos_res['test_r2'].mean():.3f} +/- {cog_hi_pos_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-cog_hi_pos_res['test_neg_mean_absolute_error'].mean():.3f} +/- {cog_hi_pos_res['test_neg_mean_absolute_error'].std():.3f}")


  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(cog_hi_pos_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, c_hi_pos_edge_c, c_emp_hi, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


cog_hi_pos_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
cog_hi_pos_errors



    ## DECISION - LPE POSITIVE COGNITIVE MODEL 
      #> although it performs better than a null model
      #> it is clear this model does not generalize so we will not test further





## COGNITIVE HIGH NEGATIVE MODEL________________________

  # adding controls
c_hi_neg_edge_c = pd.concat([pd.DataFrame(c_hi_neg_edge),pd.concat([df_hi[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==1]], axis=1).set_index(pd.DataFrame(a_hi_neg_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  cog_hi_neg_res = cross_validate(model_s, c_hi_neg_edge_c, c_emp_hi, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-cog_hi_neg_res['train_neg_mean_squared_error'].mean():.3f} +/- {cog_hi_neg_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{cog_hi_neg_res['train_r2'].mean():.3f} +/- {cog_hi_neg_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-cog_hi_neg_res['train_neg_mean_absolute_error'].mean():.3f} +/- {cog_hi_neg_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-cog_hi_neg_res['test_neg_mean_squared_error'].mean():.3f} +/- {cog_hi_neg_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{cog_hi_neg_res['test_r2'].mean():.3f} +/- {cog_hi_neg_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-cog_hi_neg_res['test_neg_mean_absolute_error'].mean():.3f} +/- {cog_hi_neg_res['test_neg_mean_absolute_error'].std():.3f}")




  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(cog_hi_neg_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, c_hi_neg_edge_c, c_emp_hi, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


cog_hi_neg_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
cog_hi_neg_errors



    ## DECISION - HIGH COGNIRTIVE NEGATIVEI MODEL
      #> generalizes and fits better than a null model - good 



  ## PERMUTATION TESTING 

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(c_hi_neg_edge_c, c_emp_hi)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    c_hi_neg_edge_c, 
    c_emp_hi, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 AND MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="gold")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-3,1.25)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_cog_neg_LPE.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()



  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(c_hi_neg_edge_c, c_emp_hi,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Cognitive Empathy": test_y, "Predicted Cognitive Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Cognitive Empathy", y="Predicted Cognitive Empathy",
                color="gold", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Negative Connectivity in Low Prosocial Emotion Specifier')
plt.text(20,6,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_cog_neg_LPE.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()





                    #### aff EMP - LOW CU ####

## Affective LOW positive model____________________________________

  # adding controls
a_lo_pos_edge_c = pd.concat([pd.DataFrame(a_lo_pos_edge),pd.concat([df_lo[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==0]], axis=1).set_index(pd.DataFrame(a_lo_pos_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  aff_lo_pos_res = cross_validate(model_s, a_lo_pos_edge_c, a_emp_lo, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-aff_lo_pos_res['train_neg_mean_squared_error'].mean():.3f} +/- {aff_lo_pos_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{aff_lo_pos_res['train_r2'].mean():.3f} +/- {aff_lo_pos_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-aff_lo_pos_res['train_neg_mean_absolute_error'].mean():.3f} +/- {aff_lo_pos_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-aff_lo_pos_res['test_neg_mean_squared_error'].mean():.3f} +/- {aff_lo_pos_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{aff_lo_pos_res['test_r2'].mean():.3f} +/- {aff_lo_pos_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-aff_lo_pos_res['test_neg_mean_absolute_error'].mean():.3f} +/- {aff_lo_pos_res['test_neg_mean_absolute_error'].std():.3f}")



  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(aff_lo_pos_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, a_lo_pos_edge_c, a_emp_lo, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


aff_lo_pos_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
aff_lo_pos_errors




  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(a_lo_pos_edge_c, a_emp_lo)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    a_lo_pos_edge_c, 
    a_emp_lo, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 ADN MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="orange")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-0.85,0.75)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_aff_pos.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()


  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(a_lo_pos_edge_c, a_emp_lo,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="orange", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Positive Connectivity in Normative')
plt.text(23.5,14,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_aff_pos.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()





## AFFECTOVE LOW NEGATIVE MODEL_______________________________________ 

  # adding controls
a_lo_neg_edge_c = pd.concat([pd.DataFrame(a_lo_neg_edge),pd.concat([df_lo[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==0]], axis=1).set_index(pd.DataFrame(a_lo_neg_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  aff_lo_neg_res = cross_validate(model_s, a_lo_neg_edge_c, a_emp_lo, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-aff_lo_neg_res['train_neg_mean_squared_error'].mean():.3f} +/- {aff_lo_neg_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{aff_lo_neg_res['train_r2'].mean():.3f} +/- {aff_lo_neg_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-aff_lo_neg_res['train_neg_mean_absolute_error'].mean():.3f} +/- {aff_lo_neg_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-aff_lo_neg_res['test_neg_mean_squared_error'].mean():.3f} +/- {aff_lo_neg_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{aff_lo_neg_res['test_r2'].mean():.3f} +/- {aff_lo_neg_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-aff_lo_neg_res['test_neg_mean_absolute_error'].mean():.3f} +/- {aff_lo_neg_res['test_neg_mean_absolute_error'].std():.3f}")




  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(aff_lo_neg_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, a_lo_neg_edge_c, c_emp_lo, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


aff_lo_neg_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
aff_lo_neg_errors




  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(a_lo_neg_edge_c, a_emp_lo)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    a_lo_neg_edge_c, 
    a_emp_lo, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 ADN MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="purple")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-1,1)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_aff_neg.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()


  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(a_lo_neg_edge_c, a_emp_lo,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="purple", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Negative Connectivity in Normative')
plt.text(24,14,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_aff_neg.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()







                  #### ML MODEL: COG EMP - NO LPE ####

## COGNITIVE LOW POSITIVE MODEL__________________________________

  # adding controls
c_lo_pos_edge_c = pd.concat([pd.DataFrame(c_lo_pos_edge),pd.concat([df_lo[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==0]], axis=1).set_index(pd.DataFrame(c_lo_pos_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  cog_lo_pos_res = cross_validate(model_s, c_lo_pos_edge_c, c_emp_lo, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-cog_lo_pos_res['train_neg_mean_squared_error'].mean():.3f} +/- {cog_lo_pos_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{cog_lo_pos_res['train_r2'].mean():.3f} +/- {cog_lo_pos_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-cog_lo_pos_res['train_neg_mean_absolute_error'].mean():.3f} +/- {cog_lo_pos_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-cog_lo_pos_res['test_neg_mean_squared_error'].mean():.3f} +/- {cog_lo_pos_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{cog_lo_pos_res['test_r2'].mean():.3f} +/- {cog_lo_pos_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-cog_lo_pos_res['test_neg_mean_absolute_error'].mean():.3f} +/- {cog_lo_pos_res['test_neg_mean_absolute_error'].std():.3f}")


  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(cog_lo_pos_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, c_lo_pos_edge_c, c_emp_lo, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


cog_lo_pos_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
cog_lo_pos_errors




  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(c_lo_pos_edge_c, c_emp_lo)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    c_lo_pos_edge_c, 
    c_emp_lo, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))


    # PLOTTING PERMUTED R2 AND MODEL PREDICTION
      # Permuted R2 distribition
perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show, savefig

fig, ax = plt.subplots()
# ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, x= "perm_scores", color="red")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
plt.xlim(-1.5,1)
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", horizontalalignment='left', size='large', color='black', weight='semibold')
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_hist_cog_pos.tiff", dpi=700)
plt.tight_layout(), plt.show(), plt.close()



  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(c_lo_pos_edge_c, c_emp_lo,test_size= .50, random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True Cognitive Empathy": test_y, "Predicted Cognitive Empathy": target_predicted})


sns.regplot(data=predicted_actual,
                x="True Cognitive Empathy", y="Predicted Cognitive Empathy",
                color="red", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title='Positive Connectivity in Normative')
plt.text(18,8,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\perm_regplot_cog_pos.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()






## COGNITIVE LOW NEGATIVE MODEL_________________________________________
  # adding controls
c_lo_neg_edge_c = pd.concat([pd.DataFrame(c_lo_neg_edge),pd.concat([df_lo[["sex", "tanner", "race", "CBCL_EXTERNALIZING_RAW"]], head_m_ave.loc[df['hi_all_rc'] ==0]], axis=1).set_index(pd.DataFrame(c_lo_neg_edge).index)],axis=1)

  # ML model
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  cog_lo_neg_res = cross_validate(model_s, c_lo_neg_edge_c, c_emp_lo, cv=outer_cv, scoring={'neg_mean_squared_error','r2','neg_mean_absolute_error'}, return_train_score=True)


  # Training data score
print(f"The mean train MSE using nested cross-validation is: "
      f"{-cog_lo_neg_res['train_neg_mean_squared_error'].mean():.3f} +/- {cog_lo_neg_res['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{cog_lo_neg_res['train_r2'].mean():.3f} +/- {cog_lo_neg_res['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-cog_lo_neg_res['train_neg_mean_absolute_error'].mean():.3f} +/- {cog_lo_neg_res['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score
print(f"The mean test MSE using nested cross-validation is: "
      f"{-cog_lo_neg_res['test_neg_mean_squared_error'].mean():.3f} +/- {cog_lo_neg_res['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{cog_lo_neg_res['test_r2'].mean():.3f} +/- {cog_lo_neg_res['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-cog_lo_neg_res['test_neg_mean_absolute_error'].mean():.3f} +/- {cog_lo_neg_res['test_neg_mean_absolute_error'].std():.3f}")


  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(cog_lo_neg_res['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, c_lo_pos_edge_c, c_emp_lo, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


cog_lo_neg_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
cog_lo_neg_errors





  ## PERMUTATION TESTING

    ## setting up model
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), l1_ratio= np.logspace(-1.5, 0, num=20), cv=inner_cv).fit(c_lo_neg_edge_c, c_emp_lo)
mm = make_pipeline(ElasticNet(alpha=tuning.alpha_, l1_ratio = tuning.l1_ratio_))


    ## permutation test with model
from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    c_lo_neg_edge_c, 
    c_emp_lo, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
    # True R2
print("permuted R2 =", score_dat)

    # permuted p-value
print("permuted p value =",perm_pvalue, "\n", 
    "rounded permuted p = " ,round(perm_pvalue,3))



      ## Model doesnt generalize or fit so we wont report on it




                #### generalizing models ####


    # generalizing models of emapthy for normative onto those with LPE



# COGNITIVE EMPATHY - POSITIVE
  # model                      
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 0].T, c_emp_lo)
zz = []
for i in range(0, fc_mat[df['hi_all_rc'] == 1].shape[0]):
  zz.append(fc_mat[df['hi_all_rc'] == 1][i][posedges == 1])
zz = np.array(zz)

train_x, test_x, train_y, test_y = train_test_split(c_lo_pos_edge, c_emp_lo ,test_size= .50, random_state=25)

target_predicted = mm.fit(train_x, train_y).predict(zz)
predicted_actual = pd.DataFrame({"True Cognitive Empathy": test_y, "Predicted Cognitive Empathy": target_predicted})


  # plot
sns.regplot(data=predicted_actual,
                x="True Cognitive Empathy", y="Predicted Cognitive Empathy",
                color="blue", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title="Generlize Positive")
plt.text(20,8.5,"".join(['R$^2$= ', str(round(mm.score(zz, test_y),3)),"$^{ns}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')
sns.despine()
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\generlize_cog_pos.tiff", dpi=700)
plt.tight_layout(),plt.show(), plt.close()



# COGNITIVE EMPATHY - NEGATIVE
  # model
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 0].T, c_emp_lo)
zz = []
for i in range(0, fc_mat[df['hi_all_rc'] == 1].shape[0]):
  zz.append(fc_mat[df['hi_all_rc'] == 1][i][negedges == 1])
zz = np.array(zz)

train_x, test_x, train_y, test_y = train_test_split(c_lo_neg_edge, c_emp_lo ,test_size= .50, random_state=25)

target_predicted = mm.fit(train_x, train_y).predict(zz)
predicted_actual = pd.DataFrame({"True Cognitive Empathy": test_y, "Predicted Cognitive Empathy": target_predicted})

  # plot
sns.regplot(data=predicted_actual,
                x="True Cognitive Empathy", y="Predicted Cognitive Empathy",
                color="green", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title="Generlize Negative")
plt.text(20,11,"".join(['R$^2$= ', str(round(mm.score(zz, c_emp_hi),3)),"$^{ns}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')
sns.despine()
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\generlize_cog_neg.tiff", dpi=700)
plt.tight_layout(),plt.show(), plt.close()




# AFFECTIVE EMPATHY - POSITIVE
  # model
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 0].T, a_emp_lo)
zz = []
for i in range(0, fc_mat[df['hi_all_rc'] == 1].shape[0]):
  zz.append(fc_mat[df['hi_all_rc'] == 1][i][posedges == 1])
zz = np.array(zz)

train_x, test_x, train_y, test_y = train_test_split(a_lo_pos_edge, a_emp_lo ,test_size= .50, random_state=25)

target_predicted = mm.fit(train_x, train_y).predict(zz)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})

  # plot
sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="orange", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title="Generlize Positive")
plt.text(24,17.5,"".join(['R$^2$= ', str(round(mm.score(zz, a_emp_hi),3)),"$^{ns}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')
sns.despine()
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\generlize_aff_pos.tiff", dpi=700)
plt.tight_layout(),plt.show(), plt.close()


# AFFECTIVE EMPATHY NEGATIVE
  # model
fit_pos,fit_neg,posedges,negedges = train_cpm(fc_mat[df['hi_all_rc'] == 0].T, a_emp_lo)
zz = []
for i in range(0, fc_mat[df['hi_all_rc'] == 1].shape[0]):
  zz.append(fc_mat[df['hi_all_rc'] == 1][i][negedges == 1])
zz = np.array(zz)

train_x, test_x, train_y, test_y = train_test_split(a_lo_neg_edge, a_emp_lo ,test_size= .50, random_state=25)

target_predicted = mm.fit(train_x, train_y).predict(zz)
predicted_actual = pd.DataFrame({"True Affective Empathy": test_y, "Predicted Affective Empathy": target_predicted})

  # plot
sns.regplot(data=predicted_actual,
                x="True Affective Empathy", y="Predicted Affective Empathy",
                color="purple", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1).set(title="Generlize Negative")
plt.text(24,17.5,"".join(['R$^2$= ', str(round(mm.score(zz, a_emp_hi),3)),"$^{ns}$"]), horizontalalignment='left', size='large', color='black', weight='semibold')
sns.despine()
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\generlize_aff_neg.tiff", dpi=700)
plt.tight_layout(),plt.show(), plt.close()






            ### GETTING REGIONS NETWORK INVOLVEMENT ####
            
    # We are downloading the DIFUMO translator between atlases to obtain
      # the networks for each region in the harvard oxford atlas
 
# DOWNLOADING TRANSLATION LIST FROM DIFUMO
difumo = pd.read_csv("https://raw.githubusercontent.com/Parietal-INRIA/DiFuMo/master/region_labeling/1024.csv")

  # gettting info on it
# difumo.info()
# pd.Series(difumo.iloc[:,7]).value_counts()

  # keeping only the columns I need
    # harvard osford atlas and noe 7 networks columns 
netwrks = difumo.iloc[:,[1,7]]

# RENAMING NETWORK VALUES TO FIGURE FRIENDLY NAMES
rr = []
for i in difumo.iloc[:,7]:
  if i == "DefaultB":
    rr.append(i.replace("DefaultB", "DMN"))
  elif i == "SalVentAttnA":
    rr.append(i.replace("SalVentAttnA", "SAL"))
  elif i == "LimbicA":
    rr.append(i.replace("LimbicA", "Limbic"))
  elif i == "SomMotA":
    rr.append(i.replace("SomMotA", "SMN"))
  elif i == "VisCent":
    rr.append(i.replace("VisCent", "Occipital"))
  elif i == "ContA":
    rr.append(i.replace("ContA", "ECN"))
  elif i == "DorsAttnB":
    rr.append(i.replace("DorsAttnB", "ECN"))
  elif i == "Cerebellar":
    rr.append(i.replace("Cerebellar", "Cerebellar"))
  else:
    rr.append("None")

netwrks['yeo_networks7'] = rr
# len(netwrks['yeo_networks7']) # 1024

# REFORMATTING HARVARD OSFORD LABELS TO MATCH
  # Now to reformat harvad oxford labels so they match 
    # keping only strings inside of the parentheses
nn = []
for i in labels:
  nn.append(i[i.find('(')+1:i.find(')')])

pd.Series(nn)

  # removing "Right"
nn2 = []
for i in nn:
  nn2.append(i.removesuffix(" Right"))

pd.Series(nn2)

  # removing "Left"
nn3 = []
for i in nn2:
  nn3.append(i.removesuffix(" Left"))

pd.Series(nn3)
# nn3[49]


  # changing for mat of 48 adn 49 to fit
nn4 = []
for i in nn3:
  nn4.append(i.replace("-formerly Supplementary Motor Cortex-", "(formerly Supplementary Motor Cortex)"))

pd.Series(nn4)


  # removing space at the end
nn5 = []
for i in nn4:
  nn5.append(i.removesuffix(" "))

pd.Series(nn5)


  # removing atlas prefix
nn6 = []
for i in nn5:
  nn6.append(i.removeprefix("atlas."))

pd.Series(nn6)


  # fixing brain stem 
nn6[105] = 'Brain-Stem'

  # correcting to match
nn6[83] = "Heschl's Gyrus (includes H1 and H2)"
nn6[84] = "Heschl's Gyrus (includes H1 and H2)"

len(nn6) ==164 # True



# matching names to networks
net_name = []
for i in nn6:
  # print(i)
  if len(np.where(netwrks.iloc[:,0]== i)[0]) >0:
    net_name.append(netwrks.iloc[:,1][list(np.where(netwrks.iloc[:,0]== i)[0])].value_counts().index[0])
  else:
    net_name.append("None")

pd.Series(net_name)

len(net_name) == 164 # True


# Adding the missing names
  # test to make sure it is workign as intended. 
len(np.concatenate([np.repeat("DMN",4),  
    np.repeat("SMN",3),
    np.repeat("Occipital",4),
    np.repeat("SAL",7),
    np.repeat("ECN",4),
    np.repeat("ECN",4),
    np.repeat("SMN",4),
    np.repeat("Cerebellar",2)])) == len(labels[132:164])


net_name[132:164] = np.concatenate([np.repeat("DMN",4),  
    np.repeat("SMN",3),
    np.repeat("Occipital",4),
    np.repeat("SAL",7),
    np.repeat("ECN",4),
    np.repeat("ECN",4),
    np.repeat("SMN",4),
    np.repeat("Cerebellar",2)])

len(net_name) == 164 # True

net_name[99:103] = np.concatenate([np.repeat("DMN",2), 
    np.repeat("SAL",2)])

len(net_name) == 164 # True

net_name[106:123] = np.repeat("Cerebellar",len(net_name[106:123]))
# pd.Series(labels)

len(net_name) == 164 # True

# NUMBER OF REGIONS IN EACH NETWORK
pd.Series(net_name).value_counts()





                    #### Number of connections ####

    # here we are identifying the number of connections for the netwokrs
    # helping to predict empathy in each of the groups to examine which 
    # networks werew involved for each 

# MAKING PEARSON FUNCTIONAL CONNECTOME
from nilearn import connectome
connectivity2 = connectome.ConnectivityMeasure(kind="correlation", vectorize=False, discard_diagonal=True) 

fc_mat2 = connectivity2.fit_transform(time_s)
  # testing to ensure correct fc size
fc_mat2[0].shape == (164,164) # True



# SETTING UP NUMBER OF CONNECTION (DENSITY) MATRICIES 

con_mat = []
for i in fc_mat2:
  con_mat.append((i < -0.3)*1 + (i > 0.3)*1)

# subsetting LPE and normative 
hi_con = pd.DataFrame(np.sum(np.array(con_mat)[df['hi_all_rc'] ==1], axis= 0), columns = net_name, index = net_name)
lo_con = pd.DataFrame(np.sum(np.array(con_mat)[df['hi_all_rc'] ==0], axis= 0), columns = net_name, index = net_name)


# LPE participants 

  # Cognitive *positive* = LPE
    # since thsi model fit we are noly retaining the indicies for later analysis
c_hi_pos_con = hi_con.iloc[np.concatenate([c_hi_pos_indicies[0],c_hi_pos_indicies[1]]),np.concatenate([c_hi_pos_indicies[0],c_hi_pos_indicies[1]])]


  # Cognitive *negative* - LPE
c_hi_neg_con = hi_con.iloc[np.concatenate([c_hi_neg_indicies[0],c_hi_neg_indicies[1]]),np.concatenate([c_hi_neg_indicies[0],c_hi_neg_indicies[1]])]

c_hi_neg_con.drop(labels= "None",
    axis = 0,
    inplace = True)

c_hi_neg_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

c_hi_neg_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = c_hi_neg_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(c_hi_neg_con2.iloc[:, np.where(c_hi_neg_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

c_hi_neg_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

    
a= sns.heatmap(c_hi_neg_con3, cmap="Blues", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_hi_cog_neg.tiff", dpi=700)
plt.show(), plt.close()



  # Affective *positive* - LPE

a_hi_pos_con = hi_con.iloc[np.concatenate([a_hi_pos_indicies[0],a_hi_pos_indicies[1]]),np.concatenate([a_hi_pos_indicies[0],a_hi_pos_indicies[1]])]

a_hi_pos_con.drop(labels= "None",
    axis = 0,
    inplace = True)

a_hi_pos_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

a_hi_pos_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = a_hi_pos_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(a_hi_pos_con2.iloc[:, np.where(a_hi_pos_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

a_hi_pos_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])


a=sns.heatmap(a_hi_pos_con3, cmap="Reds", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_hi_aff_pos.tiff", dpi=700)
plt.show(), plt.close()



  # Affective *negative* - LPE

a_hi_neg_con = hi_con.iloc[np.concatenate([a_hi_neg_indicies[0],a_hi_neg_indicies[1]]),np.concatenate([a_hi_neg_indicies[0],a_hi_neg_indicies[1]])]

a_hi_neg_con.drop(labels= "None",
    axis = 0,
    inplace = True)

a_hi_neg_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

a_hi_neg_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = a_hi_neg_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(a_hi_neg_con2.iloc[:, np.where(a_hi_neg_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

a_hi_neg_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])


a= sns.heatmap(a_hi_neg_con3, cmap="Blues", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_hi_aff_neg.tiff", dpi=700)
plt.show(), plt.close()



# Normative participants
  ## Cognitive *positive* - norm

c_lo_pos_con = lo_con.iloc[np.concatenate([c_lo_pos_indicies[0],c_lo_pos_indicies[1]]),np.concatenate([c_lo_pos_indicies[0],c_lo_pos_indicies[1]])]

c_lo_pos_con.drop(labels= "None",
    axis = 0,
    inplace = True)

c_lo_pos_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

c_lo_pos_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = c_lo_pos_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(c_lo_pos_con2.iloc[:, np.where(c_lo_pos_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

c_lo_pos_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])


a= sns.heatmap(c_lo_pos_con3, cmap="Reds", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_lo_cog_pos.tiff", dpi=700)
plt.show(), plt.close()




  ## Cognitive *negative* - norm
    # Since it doesnt fit we are only regaining the indicies for later analysis

c_lo_neg_con = lo_con.iloc[np.concatenate([c_lo_neg_indicies[0],c_lo_neg_indicies[1]]),np.concatenate([c_lo_neg_indicies[0],c_lo_neg_indicies[1]])]





  ## Affective *positive* - norm

a_lo_pos_con = lo_con.iloc[np.concatenate([a_lo_pos_indicies[0],a_lo_pos_indicies[1]]),np.concatenate([a_lo_pos_indicies[0],a_lo_pos_indicies[1]])]

a_lo_pos_con.drop(labels= "None",
    axis = 0,
    inplace = True)

a_lo_pos_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

a_lo_pos_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = a_lo_pos_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(a_lo_pos_con2.iloc[:, np.where(a_lo_pos_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

a_lo_pos_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = sns.heatmap(a_lo_pos_con3, cmap="Reds", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_lo_aff_pos.tiff", dpi=700)
plt.show(), plt.close()




  ## Affective *negative* - norm

a_lo_neg_con = lo_con.iloc[np.concatenate([a_lo_neg_indicies[0],a_lo_neg_indicies[1]]),np.concatenate([a_lo_neg_indicies[0],a_lo_neg_indicies[1]])]

a_lo_neg_con.drop(labels= "None",
    axis = 0,
    inplace = True)

a_lo_neg_con.drop(labels= "None",
    axis = 1,
    inplace = True)


a = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "DMN")[0],:], axis=0)/10).astype(int))
b = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "ECN")[0],:], axis=0)/10).astype(int))
c = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "SAL")[0],:], axis=0)/10).astype(int))
d = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "SMN")[0],:], axis=0)/10).astype(int))
e = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Limbic")[0],:], axis=0)/10).astype(int))
f = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Occipital")[0],:], axis=0)/10).astype(int))
g = list((np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Cerebellar")[0],:], axis=0)/10).astype(int))

a_lo_neg_con2 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=0), columns = a_lo_neg_con.columns, index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])

a = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "DMN")[0]], axis=1)/10).astype(int))
b = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "ECN")[0]], axis=1)/10).astype(int))
c = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "SAL")[0]], axis=1)/10).astype(int))
d = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "SMN")[0]], axis=1)/10).astype(int))
e = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "Limbic")[0]], axis=1)/10).astype(int))
f = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "Occipital")[0]], axis=1)/10).astype(int))
g = list((np.sum(a_lo_neg_con2.iloc[:, np.where(a_lo_neg_con.index == "Cerebellar")[0]], axis=1)/10).astype(int))

a_lo_neg_con3 = pd.DataFrame(np.stack([a,b,c,d,e,f,g], axis=1), columns = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"], index = ["DMN", "ECN", "SAL", "SMN", "Limbic", "Occipital", "Cerebellar"])


a= sns.heatmap(a_lo_neg_con3, cmap="Blues", square = True)
a.set_xticklabels(a.get_xticklabels(), rotation=35)
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\heat_lo_aff_neg.tiff", dpi=700)
plt.show(), plt.close()




          ### Network Stats ####

  # Given the heat map inferences on density - we will examine netwrok density 
  # both across the whole network and for individula networks

import bct


den_hi = []
for i in np.array(con_mat)[df['hi_all_rc'] == 1]:
  den_hi.append(bct.density_und(i)[0])


den_lo = []
for i in np.array(con_mat)[df['hi_all_rc'] == 0]:
  den_lo.append(bct.density_und(i)[0])



  # testing for differences 
from scipy.stats import ttest_ind
rng = np.random.default_rng()

ttest_ind(den_hi,
          den_lo, 
          permutations=10000,random_state=rng
          )

    ## yes those with LPE have less density that those low LPE




                  #### T-tests for Affective Density 


  ## DMN
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "DMN")[0],np.where(a_hi_neg_con.index == "DMN")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "DMN")[0],np.where(a_lo_neg_con.index == "DMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "DMN")[0],np.where(a_hi_pos_con.index == "DMN")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "DMN")[0],np.where(a_lo_pos_con.index == "DMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## ECN
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "ECN")[0],np.where(a_hi_neg_con.index == "ECN")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "ECN")[0],np.where(a_lo_neg_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "ECN")[0],np.where(a_hi_pos_con.index == "ECN")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "ECN")[0],np.where(a_lo_pos_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## SAL
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "SAL")[0],np.where(a_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "SAL")[0],np.where(a_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "SAL")[0],np.where(a_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "SAL")[0],np.where(a_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## SMN
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "SMN")[0],np.where(a_hi_neg_con.index == "SMN")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "SMN")[0],np.where(a_lo_neg_con.index == "SMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "SMN")[0],np.where(a_hi_pos_con.index == "SMN")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "SMN")[0],np.where(a_lo_pos_con.index == "SMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## Limbic 
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Limbic")[0],np.where(a_hi_neg_con.index == "Limbic")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Limbic")[0],np.where(a_lo_neg_con.index == "Limbic")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Limbic")[0],np.where(a_hi_pos_con.index == "Limbic")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Limbic")[0],np.where(a_lo_pos_con.index == "Limbic")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## Occipital
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Occipital")[0],np.where(a_hi_neg_con.index == "Occipital")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Occipital")[0],np.where(a_lo_neg_con.index == "Occipital")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Occipital")[0],np.where(a_hi_pos_con.index == "Occipital")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Occipital")[0],np.where(a_lo_pos_con.index == "Occipital")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## Cerebellar
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "Cerebellar")[0],np.where(a_hi_neg_con.index == "Cerebellar")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "Cerebellar")[0],np.where(a_lo_neg_con.index == "Cerebellar")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "Cerebellar")[0],np.where(a_hi_pos_con.index == "Cerebellar")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "Cerebellar")[0],np.where(a_lo_pos_con.index == "Cerebellar")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## DMN - ECN
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "DMN")[0],np.where(a_hi_neg_con.index == "ECN")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "DMN")[0],np.where(a_lo_neg_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "DMN")[0],np.where(a_hi_pos_con.index == "ECN")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "DMN")[0],np.where(a_lo_pos_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## DMN - SAL
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "DMN")[0],np.where(a_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "DMN")[0],np.where(a_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "DMN")[0],np.where(a_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "DMN")[0],np.where(a_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## ECN - SAL
ttest_ind(np.sum(a_hi_neg_con.iloc[np.where(a_hi_neg_con.index == "ECN")[0],np.where(a_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_neg_con.iloc[np.where(a_lo_neg_con.index == "ECN")[0],np.where(a_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(a_hi_pos_con.iloc[np.where(a_hi_pos_con.index == "ECN")[0],np.where(a_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(a_lo_pos_con.iloc[np.where(a_lo_pos_con.index == "ECN")[0],np.where(a_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )



                    #### T-tests for Cognitive Density




  ## DMN
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "DMN")[0],np.where(c_hi_neg_con.index == "DMN")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "DMN")[0],np.where(c_lo_neg_con.index == "DMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "DMN")[0],np.where(c_hi_pos_con.index == "DMN")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "DMN")[0],np.where(c_lo_pos_con.index == "DMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## ECN
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "ECN")[0],np.where(c_hi_neg_con.index == "ECN")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "ECN")[0],np.where(c_lo_neg_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "ECN")[0],np.where(c_hi_pos_con.index == "ECN")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "ECN")[0],np.where(c_lo_pos_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## SAL
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "SAL")[0],np.where(c_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "SAL")[0],np.where(c_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "SAL")[0],np.where(c_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "SAL")[0],np.where(c_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## SMN
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "SMN")[0],np.where(c_hi_neg_con.index == "SMN")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "SMN")[0],np.where(c_lo_neg_con.index == "SMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "SMN")[0],np.where(c_hi_pos_con.index == "SMN")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "SMN")[0],np.where(c_lo_pos_con.index == "SMN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## Limbic 
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Limbic")[0],np.where(c_hi_neg_con.index == "Limbic")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "Limbic")[0],np.where(c_lo_neg_con.index == "Limbic")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "Limbic")[0],np.where(c_hi_pos_con.index == "Limbic")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Limbic")[0],np.where(c_lo_pos_con.index == "Limbic")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## Occipital
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Occipital")[0],np.where(c_hi_neg_con.index == "Occipital")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "Occipital")[0],np.where(c_lo_neg_con.index == "Occipital")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "Occipital")[0],np.where(c_hi_pos_con.index == "Occipital")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Occipital")[0],np.where(c_lo_pos_con.index == "Occipital")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## Cerebellar
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "Cerebellar")[0],np.where(c_hi_neg_con.index == "Cerebellar")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "Cerebellar")[0],np.where(c_lo_neg_con.index == "Cerebellar")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "Cerebellar")[0],np.where(c_hi_pos_con.index == "Cerebellar")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "Cerebellar")[0],np.where(c_lo_pos_con.index == "Cerebellar")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## DMN - ECN
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "DMN")[0],np.where(c_hi_neg_con.index == "ECN")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "DMN")[0],np.where(c_lo_neg_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "DMN")[0],np.where(c_hi_pos_con.index == "ECN")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "DMN")[0],np.where(c_lo_pos_con.index == "ECN")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


  ## DMN - SAL
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "DMN")[0],np.where(c_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "DMN")[0],np.where(c_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "DMN")[0],np.where(c_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "DMN")[0],np.where(c_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )

  ## ECN - SAL
ttest_ind(np.sum(c_hi_neg_con.iloc[np.where(c_hi_neg_con.index == "ECN")[0],np.where(c_hi_neg_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_neg_con.iloc[np.where(c_lo_neg_con.index == "ECN")[0],np.where(c_lo_neg_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )


ttest_ind(np.sum(c_hi_pos_con.iloc[np.where(c_hi_pos_con.index == "ECN")[0],np.where(c_hi_pos_con.index == "SAL")[0]], axis= 1),
          np.sum(c_lo_pos_con.iloc[np.where(c_lo_pos_con.index == "ECN")[0],np.where(c_lo_pos_con.index == "SAL")[0]], axis= 1), 
          permutations=10000,random_state=rng
          )







          #### coordinates, network names, and # of connections for each region ####

  # these are for making the brain figures

# getting the list of coordinates
hi_coords = pd.concat([coords.iloc[np.concatenate([c_hi_neg_indicies[0],c_hi_neg_indicies[0]])].drop_duplicates(),
    coords.iloc[np.concatenate([a_hi_pos_indicies[0],a_hi_pos_indicies[0]])].drop_duplicates(),
    coords.iloc[np.concatenate([a_hi_neg_indicies[0],a_hi_neg_indicies[0]])].drop_duplicates()], axis = 0).drop_duplicates()

lo_coords = pd.concat([coords.iloc[np.concatenate([c_lo_pos_indicies[0],c_lo_pos_indicies[0]])].drop_duplicates(),
    coords.iloc[np.concatenate([a_lo_pos_indicies[0],a_lo_pos_indicies[0]])].drop_duplicates(),
    coords.iloc[np.concatenate([a_lo_neg_indicies[0],a_lo_neg_indicies[0]])].drop_duplicates()], axis = 0).drop_duplicates()
# getting list of names
hi_names = np.array(pd.DataFrame(net_name).iloc[hi_coords.index][0].T)
lo_names = np.array(pd.DataFrame(net_name).iloc[lo_coords.index][0].T)


# Removing 'None's
hi_coords = hi_coords.drop(hi_coords.index[np.where(hi_names == 'None')[0]])
lo_coords = lo_coords.drop(lo_coords.index[np.where(lo_names == 'None')[0]])

hi_names = np.delete(hi_names, np.where(hi_names == 'None')[0])
lo_names = np.delete(lo_names, np.where(lo_names == 'None')[0])

  # testing to make sure these line up
len(hi_names) == len(hi_coords)
len(lo_names) == len(lo_coords)

# getting density 
hi_con_fig = (np.sum(hi_con.iloc[:,hi_coords.index], axis = 0)/10).astype(int)
lo_con_fig = (np.sum(lo_con.iloc[:,lo_coords.index], axis = 0)/10).astype(int)

  # testing to make sure they match
len(hi_con_fig) == len(hi_coords)
len(lo_con_fig) == len(lo_coords)


# writing csvs for figs
# hi_coords.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\hi_coords.csv", index=False)
# lo_coords.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\lo_coords.csv", index=False)
# 
# 
# pd.DataFrame(hi_names).to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\hi_names.csv", index=False, header = False)
# pd.DataFrame(lo_names).to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\lo_names.csv", index=False, header = False)
# 
# 
# hi_con_fig.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\hi_con_fig.csv", index=False, header = False)
# lo_con_fig.to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\lo_con_fig.csv", index=False, header = False)



# pd.DataFrame(hi_con.iloc[hi_coords.index,hi_coords.index]).to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\hi_con_mat.csv", index=False, header = False)
# 
# pd.DataFrame(lo_con.iloc[lo_coords.index,lo_coords.index]).to_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\conn_pred_emp_ML\figures\lo_con_mat.csv", index=False, header = False)
# 


