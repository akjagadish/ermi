# , compare_data_statistics, compare_inputfeatures, plot_frequency_tasklabels, compare_stats_across_models
from plots import plot_decisionmaking_data_statistics
from utils import save_real_data_openML, save_real_data_lichtenberg2017


# extract real world data
save_real_data_openML(k=4, num_points=20)
save_real_data_lichtenberg2017(k=4, num_points=20)
save_real_data_lichtenberg2017(k=4, num_points=20, method='random')
save_real_data_openML(k=4, num_points=20, method='random')

# plot data statistics for 2D data
plot_decisionmaking_data_statistics(0, dim=2, condition='unknown')
plot_decisionmaking_data_statistics(2, dim=2, condition='real')
plot_decisionmaking_data_statistics(1, dim=2, condition='synthetic')

# plot data statistics for 4D data
plot_decisionmaking_data_statistics(0, dim=4, condition='ranked')
plot_decisionmaking_data_statistics(0, dim=4, condition='direction')
plot_decisionmaking_data_statistics(0, dim=4, condition='unknown')
plot_decisionmaking_data_statistics(
    2, dim=4, condition='openML', method='random')
plot_decisionmaking_data_statistics(
    2, dim=4, condition='lichtenberg2017', method='random')

# plot_frequency_tasklabels(
#    'claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5', feature_names = False, pairs = False)
# plot_frequency_tasklabels(
#     'claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5', feature_names=True, pairs=False)
# plot_frequency_tasklabels(
#     'claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5', feature_names=False, pairs=False)
# plot_frequency_tasklabels(
#     'claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5', feature_names=True, pairs=False)
# plot_frequency_tasklabels(
#     'claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5', feature_names=False, pairs=False)
# plot_frequency_tasklabels(
#     'claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5', feature_names=True, pairs=False)

# # rebuttals
# compare_data_statistics([2, 0])
# compare_data_statistics([2, 1])
# compare_data_statistics([2, 3])
# compare_inputfeatures([2, 0,])
# compare_inputfeatures([2, 0, 1, 3])
# compare_stats_across_models([2, 0, 1, 3], feature='input_features')
# compare_stats_across_models([2, 0, 1, 3], feature='input_correlation')
# compare_stats_across_models([2, 0, 1, 3], feature='gini_coefficient')
# compare_stats_across_models([2, 0, 1, 3], feature='posterior_logprob')
# compare_stats_across_models([2, 0, 1, 3], feature='performance')
# # slides
# compare_stats_across_models([2, 0, 1], feature='input_features')
# compare_stats_across_models([2, 0, 1], feature='input_correlation')
# compare_stats_across_models([2, 0, 1], feature='gini_coefficient')
# compare_stats_across_models([2, 0, 1], feature='posterior_logprob')
# compare_stats_across_models([2, 0, 1], feature='performance')
