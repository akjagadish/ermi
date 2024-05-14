from wordcloud import WordCloud
from collections import Counter
import torch.nn.functional as F
from groupBMC.groupBMC import GroupBMC
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
SYS_PATH = '/u/ajagadish/ermi'
sys.path.append(f"{SYS_PATH}/decisionmaking/")
# sys.path.append(f"{SYS_PATH}/categorisation/rl2")
sys.path.append(f"{SYS_PATH}/decisionmaking/data")
# from evaluate import evaluate_metalearner
FONTSIZE = 20


def plot_decisionmaking_data_statistics(mode=0, dim=4, condition='unkown'):

    from sklearn.preprocessing import PolynomialFeatures
    import statsmodels.api as sm

    def gini_compute(x):
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad/np.mean(x)
        return 0.5 * rmad

    def calculate_bic(n, rss, num_params):
        bic = n * np.log(rss/n) + num_params * np.log(n)
        # if bic is an array return item else return bic
        return bic.item() if isinstance(bic, np.ndarray) else bic

    def return_data_stats(data, poly_degree=2, first=False, dim=dim, include_bias=True):

        df = data.copy()
        max_tasks = 400
        max_trial = 20
        # sample tasks
        tasks = range(0, max_tasks) if first else np.random.choice(
            df.task_id.unique(), max_tasks, replace=False)
        all_corr, all_bics_linear, all_bics_quadratic, gini_coeff, all_accuraries_linear, all_accuraries_polynomial = [], [], [], [], [], []
        sign_coeff, direction_coeff = [], []
        all_features_without_norm, all_features_with_norm, all_targets_with_norm = np.array(
            []), np.array([]), np.array([])
        for i in tasks:
            df_task = df[df['task_id'] == i]
            if len(df_task) > 0:  # arbitary data size threshold
                y = df_task['target'].to_numpy()
                X = df_task["input"].to_numpy()
                X = np.stack(X)
                X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0) + 1e-6)
                y = (y - y.min(axis=0))/(y.max(axis=0) - y.min(axis=0) + 1e-6)

                # all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 0], X[:, 3])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 1], X[:, 3])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 2], X[:, 3])[0, 1])
                all_corr.append(np.corrcoef(X.T)[np.triu_indices(dim, k=1)])

                all_features_with_norm = np.concatenate(
                    [all_features_with_norm, X.flatten()])
                all_targets_with_norm = np.concatenate(
                    [all_targets_with_norm, y.flatten()])

                if (y == 0).all() or (y == 1).all():
                    pass
                else:
                    # X_linear = PolynomialFeatures(1).fit_transform(X)
                    X_linear = PolynomialFeatures(
                        1, include_bias=include_bias).fit_transform(X)

                    # # linear regression from X_linear to y
                    # linear_regresion = sm.OLS(y, X_linear).fit()

                    # # polinomial regression from X_poly
                    # X_poly = PolynomialFeatures(
                    #     poly_degree, interaction_only=True, include_bias=False).fit_transform(X)
                    # polynomial_regression = sm.OLS(y, X_poly).fit()

                    # gini coefficient from linear regression coefficients
                    params = sm.OLS(y, X_linear).fit().params
                    gini_coeff.append(gini_compute(
                        np.abs(params[1 if include_bias else 0:])))

                    per_feature_params = np.zeros((dim))
                    for i in range(dim):
                        per_feature_params[i] = sm.OLS(
                            y, X_linear[:, [0, i+1]]).fit().params[1 if include_bias else 0]

                    # sign of the coefficients
                    sign_coeff.append(np.sign(per_feature_params))

                    # direction of the coefficients
                    direction_coeff.append(per_feature_params)

                    # fit gaussian process with linear kernel to X_linear and y
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import RBF, DotProduct

                    GP_linear = GaussianProcessRegressor(
                        kernel=1.0 * DotProduct(), n_restarts_optimizer=10)
                    GP_linear.fit(X_linear, y)

                    # fit gaussian process with rbf kernel to X_poly and y
                    GP_quadratic = GaussianProcessRegressor(
                        kernel=1.0 * RBF(), n_restarts_optimizer=10)
                    GP_quadratic.fit(X_linear, y)

                    rss = np.sum((y - GP_linear.predict(X_linear))**2)
                    all_bics_linear.append(calculate_bic(
                        len(y), rss, 1))  # linear_regresion.bic)

                    rss = np.sum((y - GP_quadratic.predict(X_linear))**2)
                    all_bics_quadratic.append(calculate_bic(
                        X_linear.shape[0], rss, len(GP_quadratic.kernel.theta)))
                    # polynomial_regression.bic)

                    if X.shape[0] < max_trial:
                        pass
                    else:
                        task_accuraries_linear = []
                        # task_accuraries_polynomial = []
                        for trial in range(max_trial):
                            X_linear_uptotrial = X_linear[:trial]
                            # X_poly_uptotrial = X_poly[:trial]
                            y_uptotrial = y[:trial]

                            if (y_uptotrial == 0).all() or (y_uptotrial == 1).all() or trial == 0:
                                task_accuraries_linear.append(1.)
                                # task_accuraries_polynomial.append(0.5)
                            else:

                                # linear regression prediction
                                linear_reg = sm.OLS(
                                    y_uptotrial, X_linear_uptotrial).fit()
                                # log_reg_quadratic = sm.OLS(y_uptotrial, X_poly_uptotrial).fit(method='bfgs', maxiter=10000, disp=0)

                                y_linear_trial = linear_reg.predict(
                                    X_linear[trial])
                                # y_poly_trial = log_reg_quadratic.predict(X_poly[trial])

                                # mean squared error
                                task_accuraries_linear.append(
                                    float((y_linear_trial - y[trial]).item())**2)
                            # task_accuraries_polynomial.append(float((y_poly_trial.round() == y[trial]).item()))

                    all_accuraries_linear.append(task_accuraries_linear)
                    # all_accuraries_polynomial.append(task_accuraries_polynomial)
        all_accuraries_linear = np.array(all_accuraries_linear).mean(0)
        # all_accuraries_polynomial = np.array(all_accuraries_polynomial).mean(0)

        logprobs = torch.from_numpy(-0.5 *
                                    np.stack((all_bics_linear, all_bics_quadratic), -1))
        joint_logprob = logprobs + \
            torch.log(torch.ones([]) / logprobs.shape[1])
        marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
        posterior_logprob = joint_logprob - marginal_logprob

        return all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff

    # set env_name and color_stats based on mode
    if mode == 0:
        if dim == 2:
            env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_{condition}'
        elif dim == 4:
            num_tasks = 8770 if condition == 'ranked' else 8220
            # env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks{num_tasks}_run0_procid0_pversion2_{condition}'
            env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks{num_tasks}_run0_procid1_pversion{condition}'
        color_stats = '#405A63'  # '#2F4A5A'# '#173b4f'
    elif mode == 1:  # last plot
        env_name = f'{SYS_PATH}/decisionmaking/data/synthetic_decisionmaking_tasks_dim2_data20_tasks10000'
        color_stats = '#66828F'  # 5d7684'# '#5d7684'
    elif mode == 2:  # first plot
        env_name = f'{SYS_PATH}/decisionmaking/data/real_data_dim{dim}'
        color_stats = '#173b4f'  # '#0D2C3D' #'#8b9da7'
    # elif mode == 3:
    #     env_name = f'{SYS_PATH}/decisionmaking/data/synthetic_tasks_dim4_data650_tasks1000_nonlinearTrue'
    #     color_stats = '#5d7684'

    # load data
    data = pd.read_csv(f'{env_name}.csv')
    data.input = data['input'].apply(lambda x: np.array(eval(x)))
    if mode == 2 or mode == 1:
        data.target = data['target'].apply(lambda x: np.array(eval(x)))
        # TODO: shuffle order of input features (but it is artifiically inducing lack of ranking)
        data.input = data.input.apply(np.random.permutation)

    if os.path.exists(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz'):
        stats = np.load(
            f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz', allow_pickle=True)
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear = stats['all_corr'], stats[
            'gini_coeff'], stats['posterior_logprob'], stats['all_accuraries_linear']
        all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff = stats[
            'all_accuraries_polynomial'], stats['all_targets_with_norm'], stats['all_features_with_norm'], stats['sign_coeff'], stats['direction_coeff']
    else:
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, \
            all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff = return_data_stats(
                data, dim=dim)

    gini_coeff = np.array(gini_coeff)
    gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    all_corr = np.array(all_corr)
    sign_coeff = np.array(sign_coeff)
    direction_coeff = np.stack(direction_coeff)
    # posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy()

    FONTSIZE = 22  # 8
    fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    sns.histplot(all_corr.reshape(-1), ax=axs[0], bins=11, binrange=(
        -1., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(
        0., gini_coeff.max()), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(np.argmax(np.abs(direction_coeff), axis=1), ax=axs[2], bins=dim, binrange=(
        -0.5, dim-0.5), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(sign_coeff.reshape(-1), ax=axs[3], bins=3, binrange=(
        -1.5, 1.5), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)

    axs[0].set_ylim(0, .4)
    axs[1].set_ylim(0, .4)
    axs[2].set_ylim(0, .6)
    axs[3].set_ylim(0, 1.)
    # axs[3].set_ylim(0,  0.75)

    # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    # axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    axs[2].set_xticks(np.arange(0, dim, 1))
    axs[2].set_xticklabels([f"coef{i+1}" for i in range(dim)])
    axs[3].set_xticks(np.arange(-1, 2, 1))
    axs[3].set_xticklabels(['negative', 'unsigned', 'positive'])

    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axs[0].set_ylabel('Proportion', fontsize=FONTSIZE)
    axs[1].set_ylabel('', fontsize=FONTSIZE)
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    if mode == 3:
        axs[0].set_xlabel('Pearson\'s r', fontsize=FONTSIZE)
        axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
        axs[2].set_xlabel('Regression coefficient', fontsize=FONTSIZE)
        axs[3].set_xlabel('Sign of regression coefficient', fontsize=FONTSIZE)

    # set title
    if mode == 2:
        axs[0].set_title('Input Correlation', fontsize=FONTSIZE)
        axs[1].set_title('Sparsity', fontsize=FONTSIZE)
        axs[2].set_title('Ranking', fontsize=FONTSIZE)
        axs[3].set_title('Direction', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.savefig(
        f'{SYS_PATH}/figures/decisionmaking_stats_{str(mode)}_{str(dim)}_{condition}.svg', bbox_inches='tight')
    plt.show()

    # FONTSIZE = 22  # 8
    # fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    # axs[0].plot(all_accuraries_linear, color=color_stats, alpha=1., lw=3)
    # # axs[0].plot(all_accuraries_polynomial, alpha=0.7)
    # sns.histplot(np.stack(all_corr), ax=axs[1], bins=11, binrange=(
    #     0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # sns.histplot(np.stack(all_targets_with_norm).reshape(-1), ax=axs[2], bins=11, binrange=(
    #     0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # sns.histplot(posterior_logprob, ax=axs[3], bins=5, binrange=(
    #     0.0, 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # # axs[1].set_xlim(-1, 1)

    # axs[0].set_ylim(0., 1.05)
    # axs[1].set_ylim(0,  0.2)
    # axs[2].set_ylim(0,  0.2)
    # axs[2].set_xlim(0., 1.05)
    # axs[3].set_xlim(0., 1.05)

    # # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    # axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    # # set tick size
    # axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    # axs[0].set_ylabel('Squared Error', fontsize=FONTSIZE)
    # axs[1].set_ylabel('Proportion', fontsize=FONTSIZE)
    # axs[2].set_ylabel('')
    # axs[3].set_ylabel('')

    # if mode == 3:
    #     axs[0].set_xlabel('Trials', fontsize=FONTSIZE)
    #     axs[1].set_xlabel('Input', fontsize=FONTSIZE)
    #     axs[2].set_xlabel('Target', fontsize=FONTSIZE)
    #     axs[3].set_xlabel('Posterior probability ', fontsize=FONTSIZE)

    # # set title
    # if mode == 2:
    #     axs[0].set_title('Performance', fontsize=FONTSIZE)
    #     axs[1].set_title('Input distribution', fontsize=FONTSIZE)
    #     axs[2].set_title('Target distribution', fontsize=FONTSIZE)
    #     axs[3].set_title('Linearity', fontsize=FONTSIZE)

    # plt.tight_layout()
    # sns.despine()
    # plt.savefig(
    #     f'{SYS_PATH}/figures/supp_decisionmaking_stats_{str(mode)}_{str(dim)}_{condition}_test.svg', bbox_inches='tight')
    # plt.show()

    # save computed stats in one .npz file
    if not os.path.exists(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}_test.npz'):
        np.savez(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz', all_corr=all_corr, gini_coeff=gini_coeff, posterior_logprob=posterior_logprob, all_accuraries_linear=all_accuraries_linear,
                 all_accuraries_polynomial=all_accuraries_polynomial, all_targets_with_norm=all_targets_with_norm, all_features_with_norm=all_features_with_norm, sign_coeff=sign_coeff, direction_coeff=direction_coeff)


def world_cloud(file_name, path='/u/ajagadish/ermi/decisionmaking/data/synthesize_problems', feature_names=True, pairs=False, top_labels=50):

    df = pd.read_csv(f'{path}/{file_name}.csv')
    dim = int(file_name.split("_dim")[1].split("_")[0])
    df.feature_names = df['feature_names'].apply(lambda x: list(eval(x)[:dim]))

    def to_lower(ff):
        return [x.lower() for x in ff]

    # name of the column containing the feature names
    column_name = 'feature_names' if feature_names else 'target_names'
    # count of number of times a type of features occurs
    list_counts = Counter([tuple(features) for features in df[column_name]]
                          if pairs else np.stack(df[column_name].values).reshape(-1))

    # sort the Counter by counts in descending order
    sorted_list_counts = sorted(
        list_counts.items(), key=lambda x: x[1], reverse=True)

    # extract the counts and names for the top 50 labels
    task_labels = np.array([task_label[0]
                            for task_label in sorted_list_counts[:top_labels]])
    label_counts = np.array([task_label[1]
                             for task_label in sorted_list_counts[:top_labels]])
    label_names = ['-'.join(task_labels[idx])
                   for idx in range(len(task_labels))] if pairs else task_labels

    # make a dict with task labels and counts
    word_freq = {}
    for idx in range(len(label_names)):
        word_freq[label_names[idx]] = label_counts[idx]

    # generate word cloud
    # wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white').generate_from_frequencies(word_freq)
    wordcloud = WordCloud(width=1300, height=700, background_color='white', max_font_size=100,
                          collocations=False, colormap='inferno', prefer_horizontal=1).generate_from_frequencies(word_freq)
    plt.figure(figsize=(13, 7), dpi=1000)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    wordcloud.to_file(
        f'{SYS_PATH}/figures/wordcloud_{column_name}_paired={pairs}_top{top_labels}.png')
