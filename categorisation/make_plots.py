from plots import model_comparison_badham2017, model_simulations_smith1998, model_simulations_shepard1961, model_comparison_devraj2022, model_comparison_johanssen2002, plot_dataset_statistics, compare_data_statistics, compare_inputfeatures, plot_frequency_tasklabels, compare_stats_across_models

# main paper
plot_dataset_statistics(0)
plot_dataset_statistics(1)
plot_dataset_statistics(2)
plot_dataset_statistics(3)
model_simulations_shepard1961('main')
model_simulations_shepard1961('supplementary')
model_comparison_badham2017()
model_simulations_smith1998('main')
model_simulations_smith1998('supplementary')
model_comparison_devraj2022()
model_comparison_johanssen2002('main')
model_comparison_johanssen2002('supplementary')
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5', feature_names=False, pairs=False)
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5', feature_names=True, pairs=False)
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5', feature_names=False, pairs=False)
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5', feature_names=True, pairs=False)
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5', feature_names=False, pairs=False)
plot_frequency_tasklabels(
    'claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5', feature_names=True, pairs=False)

# rebuttals
compare_data_statistics([2, 0])
compare_data_statistics([2, 1])
compare_data_statistics([2, 3])
compare_inputfeatures([2, 0,])
compare_inputfeatures([2, 0, 1, 3])
compare_stats_across_models([2, 0, 1, 3], feature='input_features')
compare_stats_across_models([2, 0, 1, 3], feature='input_correlation')
compare_stats_across_models([2, 0, 1, 3], feature='gini_coefficient')
compare_stats_across_models([2, 0, 1, 3], feature='posterior_logprob')
compare_stats_across_models([2, 0, 1, 3], feature='performance')
model_simulations_shepard1961('rebuttals', num_blocks=6)
model_simulations_smith1998('rebuttals')

# slides
# compare_stats_across_models([2, 0, 1], feature='input_features')
# compare_stats_across_models([2, 0, 1], feature='input_correlation')
# compare_stats_across_models([2, 0, 1], feature='gini_coefficient')
# compare_stats_across_models([2, 0, 1], feature='posterior_logprob')
# compare_stats_across_models([2, 0, 1], feature='performance')
