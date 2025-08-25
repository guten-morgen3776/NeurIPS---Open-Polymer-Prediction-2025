## ベースラインモデルの説明
```
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

desc_names = [desc[0] for desc in Descriptors.descList]
descriptors = [compute_all_descriptors(smi) for smi in train['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)

train = pd.concat([train,descriptors],axis=1)
```
ここで元データのsmiles列に対してrdkitを適用している
```
def lgb_kfold(train_df, test_df, target, feats, folds):    
    params = {    
         'objective' : 'mae',#'binary', 
         'metric' : 'mae', 
         'num_leaves': 31,
         'min_data_in_leaf': 30,#30,
         'learning_rate': 0.01,
         'max_depth': -1,
         'max_bin': 256,
         'boosting': 'gbdt',
         'feature_fraction': 0.7,
         'bagging_freq': 1,
         'bagging_fraction': 0.7,
         'bagging_seed': 42,
         "lambda_l1":1,
         "lambda_l2":1,
         'verbosity': -1,        
         'num_boost_round' : 20000,
         'device_type' : 'cpu'        
    }      
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    df_importances = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[target])):     
        print ('n_fold:',n_fold)
        
        train_x = train_df[feats].iloc[train_idx].values
        train_y = train_df[target].iloc[train_idx].values
        
        valid_x = train_df[feats].iloc[valid_idx].values
        valid_y = train_df[target].iloc[valid_idx].values

        test_x = test_df[feats]
        
        print ('train_x',train_x.shape)
        print ('valid_x',valid_x.shape)    
        print ('test_x',test_x.shape)  
        
        dtrain = lgb.Dataset(train_x, label=train_y, )
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain, ) 
        callbacks = [
        lgb.log_evaluation(period=100,),
        lgb.early_stopping(200)    
        ]
        bst = lgb.train(params, dtrain,valid_sets=[dval,dtrain],callbacks=callbacks,
                       ) 

        #---------- feature_importances ---------#
        feature_importances = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)#[:100]
        for f in feature_importances[:30]:
            print (f)       
            
        new_feats = []
        importances = []
        for f in feature_importances:
            new_feats.append(f[0])
            importances.append(f[1])
        df_importance = pd.DataFrame()
        df_importance['feature'] = new_feats
        df_importance['importance'] = importances
        df_importance['fold'] = n_fold
        
        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        # oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        # cv_list.append(oof_cv)
        # print (cv_list)
        
        sub_preds += bst.predict(test_x, num_iteration=bst.best_iteration) / n_splits
        
        #bst.save_model(model_path+'lgb_fold_' + str(n_fold) + '.txt', num_iteration=bst.best_iteration)     

        df_importances = pd.concat([df_importances,df_importance])    
        
    # cv = mae(train_df[target],  oof_preds)
    # print (cv)
    
    return oof_preds,sub_preds
```
### まとめ（この関数の意味）
- やっていること
    - KFold で学習/検証データを分ける
    - 各 fold ごとに LightGBM を学習
    - 検証データの OOF 予測を保存
    - テストデータ予測を fold 平均する
    - 特徴量重要度を fold ごとに保存
- 最終的に得られるもの
    - OOF 予測 (oof_preds) → CV スコア算出や meta モデルに使える
    - テスト予測 (sub_preds) → 提出用
    - 特徴量重要度（df_importances） → 特徴量選択・可視化に使える
```
n_splits = 5
seed = 817
folds = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
feats = ['MaxAbsEStateIndex', 'MaxEStateIndex',
       'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
       'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
       'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
       'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI',
       'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
       'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc',
       'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
       'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
       'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
       'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
       'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
       'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
       'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
       'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
       'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
       'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
       'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
       'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
       'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
       'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
       'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
       'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
       'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
       'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles',
       'NumAromaticHeterocycles', 'NumAromaticRings',
       'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors',
       'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles',
       'NumRotatableBonds', 'NumSaturatedCarbocycles',
       'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms',
       'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP',
       'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
       'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
       'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
       'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
       'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
       'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
       'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
       'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
       'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
       'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
       'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
       'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
       'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
       'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
       'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
       'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
       'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
       'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
for t in targets:
    print (t)
    if len(test)<10:
        test[t] = 0
    else:    
        train_df = train[train[t].notnull()]
        oof_preds,sub_preds = lgb_kfold(train_df, test, t, feats, folds)
        test[t] = sub_preds
```
このコードは、
- 分子記述子を特徴量とする LightGBM 回帰モデル を構築
- ターゲットごとに独立して学習・予測（欠損値がある場合はそのターゲットについてだけ学習できるデータを使う）
- 交差検証を行って OOF と Test の予測を得る
- test データフレームに予測値を追加していく
👉 マルチターゲット問題を「ターゲットごとにシングルタスク回帰を回す」スタイルで処理しているわけです。
