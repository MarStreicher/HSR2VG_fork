import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from itertools import permutations
from collections import defaultdict
###############################################################################
def train_ML_PLSR(X_train_df, X_test_df, y_train_df, y_test_df, n_comp):
    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_df.values.reshape(-1, 1) 

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)

    model = PLSRegression(n_components=n_comp)
    model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test_df.values.ravel()

    r2 = r2_score(y_test_flat, y_pred_flat)
    r, pval = pearsonr(y_test_flat, y_pred_flat)

    return r2, r, pval
###############################################################################
rename_dict={
          'UNL_Maize': {'chl':'CHL'},
          'UNL_Sorghum': {'chl':'CHL'},
          'UNL_Soybean': {'chl':'CHL'},
          'LOPEX':{'Carb.':'C','Nit':'N'}
          }
          
traits_dict={
          'UCAM_Maize_all':['LMA', 'N','C'],
          'UNL_Maize': ['chl', 'LMA',  'Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Sorghum': ['chl', 'LMA','Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Soybean': ['chl', 'LMA','Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Camelina': ['Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'ANGERS':['CHL','CAR','EWT','LMA'],
          'LOPEX':['CHL','CAR','EWT','LMA','Nit','Carb.']
          }
          
new_traits_dict = {}
for dataset, traits in traits_dict.items():
    rename_map = rename_dict.get(dataset, {})  
    new_traits = [rename_map.get(trait, trait) for trait in traits]
    new_traits_dict[dataset] = new_traits
           
trait_to_datasets = defaultdict(list)
           
for dataset, traits in new_traits_dict.items():
    for trait in traits:
        trait_to_datasets[trait].append(dataset)
           
trait_pairs= {}
          
for trait, datasets in trait_to_datasets.items():
    if len(datasets) >= 2:
        trait_pairs[trait] = list(permutations(datasets, 2)) 
##############################################################################
real_trait_name = {}

for dataset, traits in traits_dict.items():
    rename_map = rename_dict.get(dataset, {})
    trait_map = {}
    for trait in traits:
        new_name = rename_map.get(trait, trait)
        trait_map[new_name] = trait  
    real_trait_name[dataset] = trait_map
##############################################################################
###############################################################################
start_lambda = 400
end_lambda = 2400
hsr_cols = np.arange(start_lambda, end_lambda + 1)
hsr_cols = [str(num) for num in hsr_cols]
#######################################
map_name_list = ['spectral_data','eccentricity_weighted_map','closeness_weighted_map','degree_map'] 
###############################################################################
###############################################################################
res_list = []
for trait in trait_pairs.keys():
    for sp_tuple in trait_pairs[trait]:
        sp1,sp2 = sp_tuple[0],sp_tuple[1]
        if sp1==sp2: 
            raise Exception("Source and target species are the same!") 
        sp1_trait = real_trait_name[sp1][trait]
        sp2_trait = real_trait_name[sp2][trait]
        for map_name in map_name_list:
            ########################################
            df = pd.read_csv(f'res/PLSR/PLSR_{sp1}_{map_name}_400_2400.csv')
            df = df[df['trait'].isin([sp1_trait])].copy()
            df['mean_r'] = df['mean_r'].round(2)
            df_max = df.loc[df.groupby('trait', sort=False)['mean_r'].idxmax()].copy()
            nc= df_max['n_components'].values[0]
            print(f"Trait: {trait} | Source: {sp1} | Target: {sp2} | Map: {map_name} | n_components: {nc}")
            ###########################################################################
            res_traits1 = pd.read_csv(f'res/VG_maps/{sp1}_{map_name}_400_2400.csv')
            res_traits1 = res_traits1[['ID']+[sp1_trait] + hsr_cols]
            res_traits1 = res_traits1[res_traits1[sp1_trait] >= 0]
            ########################################
            res_traits2 = pd.read_csv(f'res/VG_maps/{sp2}_{map_name}_400_2400.csv')
            res_traits2 = res_traits2[['ID']+[sp2_trait] + hsr_cols]
            res_traits2 = res_traits2[res_traits2[sp2_trait] >= 0]
            ########################################
            ########################################
            graph_traits_df1 = res_traits1.dropna(how='any')
            graph_traits_df2 = res_traits2.dropna(how='any')
            ###########################################################################
            ###########################################################################
            X1 =graph_traits_df1 [hsr_cols].copy()    
            y1 = graph_traits_df1 [[sp1_trait]].copy()
            X2 =graph_traits_df2 [hsr_cols].copy()    
            y2 = graph_traits_df2 [[sp2_trait]].copy()
            ###############################################################################
            r2,pearson_correlation,pval = train_ML_PLSR(X1, X2, y1, y2,n_comp=nc)
            print(f"R2: {r2:.3f} | Pearson r: {pearson_correlation:.3f}")
            res_list.append([trait,f'{sp1}-{sp2}',map_name,'PLSR',r2,pearson_correlation,pval])
            print('#################')
        ##############################
        print('####################################')
    ###############################################################################
res_df = pd.DataFrame(res_list, columns = ['trait', 'sp', 'map','model','R2','r','pval'])
res_df.to_csv('res/train_res_transferability_ALL_2dec_400_2400.csv',index=False)
###############################################################################
