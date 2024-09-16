import pandas as pd
df_data = pd.read_csv('/u/ajagadish/ermi/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown.csv')
df_features = pd.read_csv('/u/ajagadish/ermi/decisionmaking/data/synthesize_problems/claude_synthesized_functionlearning_problems_paramsNA_dim2_tasks9254_pversion0.csv')
df_combined = pd.read_csv('/u/ajagadish/ermi/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown.csv')

features = []
targets = []
for task_id in df_data.task_id.values:
    feature_name = eval(df_features[df_features.task_id==task_id].feature_names.values[0])
    target_name =  df_features[df_features.task_id==task_id].target_names.values
    features.append(feature_name)
    targets.append(target_name)
df_combined.insert(2, 'feature_names', features)
df_combined.insert(3, 'target_names', targets)

df_combined.to_csv('/u/ajagadish/ermi/decisionmaking/data/combined/claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown.csv')