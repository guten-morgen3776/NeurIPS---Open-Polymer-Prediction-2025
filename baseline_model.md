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
まとめ（この関数の意味）
やっていること
KFold で学習/検証データを分ける
各 fold ごとに LightGBM を学習
検証データの OOF 予測を保存
テストデータ予測を fold 平均する
特徴量重要度を fold ごとに保存
最終的に得られるもの
OOF 予測 (oof_preds) → CV スコア算出や meta モデルに使える
テスト予測 (sub_preds) → 提出用
特徴量重要度（df_importances） → 特徴量選択・可視化に使える
