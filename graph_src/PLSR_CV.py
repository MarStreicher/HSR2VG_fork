import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression  
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
###############################################################################
def pls_cross_validation_ensemble(X, y, n_folds, n_seeds, n_component):
    r_values_all_seeds = []
    r2_values_all_seeds = []

    for seed in range(0, n_seeds * 10, 10):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        r_values = []
        r2_values = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  

            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))

            pls = PLSRegression(n_components=n_component)
            pls.fit(X_train_scaled, y_train_scaled)

            y_pred_scaled = pls.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

            y_test = y_test.to_numpy().ravel()   
            
            r = stats.pearsonr(y_test, y_pred)[0]
            r2 = r2_score(y_test, y_pred)

            r_values.append(r)
            r2_values.append(r2)

        r_values_all_seeds.append(r_values)
        r2_values_all_seeds.append(r2_values)

    r2_values_all_seeds = np.array(r2_values_all_seeds)
    mean_r2_values_all_seeds = np.mean(r2_values_all_seeds)
    std_r2_values_all_seeds = np.std(r2_values_all_seeds)
    
    r_values_all_seeds = np.array(r_values_all_seeds)
    mean_r_all_seeds = np.mean(r_values_all_seeds)
    std_r_all_seeds = np.std(r_values_all_seeds)
    
    print(f"Mean R2: {mean_r2_values_all_seeds:.4f} ± {std_r2_values_all_seeds:.4f}")
    print(f"Mean r: {mean_r_all_seeds:.4f} ± {std_r_all_seeds:.4f}")
    
    return mean_r_all_seeds, std_r_all_seeds, mean_r2_values_all_seeds, std_r2_values_all_seeds

###############################################################################
###############################################################################
def plsr_cv_on_species_ensemble(species,map_name,bands_dict, traits_dict, n_folds, n_seeds, comps_ls):
    res0 = []
    trait_ls = traits_dict[species]
    start,end = bands_dict[species]
    data0=pd.read_csv(f'res/VG_maps/{species}_{map_name}_{start}_{end}.csv')
    for nc in comps_ls:
        for trait in trait_ls:
            hsr_cols=[str(i) for i in np.arange(start,end+1)]
            data = data0[hsr_cols+[trait]].copy().dropna()
            data = data[data[trait] >= 0] #### To remove unrealistic trait values 
            print(f'n_comp:{nc},species:{species},trait:{trait}')
            X,y = data[hsr_cols],data[trait]
            mean_r, std_r, mean_r2, std_r2 = pls_cross_validation_ensemble(X, y, n_folds, n_seeds, nc)
            res0.append([species,map_name,trait,start,end,nc,mean_r, std_r, mean_r2, std_r2, n_folds, n_seeds])
            print('########################################')
    res_df0 = pd.DataFrame(res0,columns=['species','map','trait','start_lambda','end_lambda','n_components','mean_r', 'std_r', 'mean_r2', 'std_r2','n_folds', 'n_seeds'])
    res_df0.to_csv(f'res/PLSR/PLSR_{species}_{map_name}_{start}_{end}.csv',index=False) 
############################################################################### 
Traits={'UCAM_Maize_all':['A_sat', 'a400', 'gsw', 'LMA', 'SL','Vpmax', 'Vmax', 'C', 'N'],
          'UNL_Maize': ['chl', 'LMA',  'Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Sorghum': ['chl', 'LMA','Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Soybean': ['chl', 'LMA','Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'UNL_Camelina': ['Mg', 'S', 'LWC', 'N', 'P', 'K', 'Ca'],
          'ANGERS':['CHL','CAR','ANT','EWT','LMA'],
           'LOPEX':['CHL','CAR','EWT','LMA','FW','DW','LT','Carb.','Nit']}

map_name_list = ['spectral_data','degree_map','degree_weighted_map','eccentricity_map','eccentricity_weighted_map'
                 ,'betweenness_map','betweenness_weighted_map','betweenness_weighted_map','closeness_map',
                'closeness_weighted_map','eigenvector_map', 'eigenvector_weighted_map'] 

hsr_range={'UCAM_Maize_all': (400,2400),'UNL_Maize': (400,2400),'UNL_Sorghum':(400,2400),
                'UNL_Soybean':(400,2400),'UNL_Camelina':(400,2400), 'Eudicot': (400,2400), 
                'LOPEX': (400,2400), 'ANGERS': (400,2400)}   
#######################################
# test
components_n_ls= np.arange(2,76,1)
sp = 'LOPEX'
hsr_map = map_name_list[1]
plsr_cv_on_species_ensemble(species=sp,map_name=hsr_map,bands_dict = hsr_range,traits_dict = Traits, n_folds=5, n_seeds=10, comps_ls=components_n_ls)
###############################################################################
