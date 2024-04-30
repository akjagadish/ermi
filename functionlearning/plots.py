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
sys.path.append(f"{SYS_PATH}/functionlearning/")
# sys.path.append(f"{SYS_PATH}/categorisation/rl2")
sys.path.append(f"{SYS_PATH}/functionlearning/data")
# from evaluate import evaluate_metalearner
FONTSIZE = 20


def plot_functionlearning_data_statistics(mode=0):

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

    def return_data_stats(data, poly_degree=2, first=False):

        df = data.copy()
        max_tasks = 400
        max_trial = 20
        # sample tasks
        tasks = range(0, max_tasks) if first else np.random.choice(
            df.task_id.unique(), max_tasks, replace=False)
        all_corr, all_bics_linear, all_bics_quadratic, gini_coeff, all_accuraries_linear, all_accuraries_polynomial = [], [], [], [], [], []
        all_features_without_norm, all_features_with_norm, all_targets_with_norm = np.array(
            []), np.array([]), np.array([])
        for i in tasks:
            df_task = df[df['task_id'] == i]
            if len(df_task) > 0:  # arbitary data size threshold
                y = df_task['target'].to_numpy()
                X = df_task["input"].to_numpy()
                X = np.stack(X)
                X = (X - X.min())/(X.max() - X.min() + 1e-6)
                y = (y - y.min())/(y.max() - y.min() + 1e-6)

                all_features_with_norm = np.concatenate(
                    [all_features_with_norm, X.flatten()])
                all_targets_with_norm = np.concatenate(
                    [all_targets_with_norm, y.flatten()])

                if (y == 0).all() or (y == 1).all():
                    pass
                else:
                    # X_linear = PolynomialFeatures(1).fit_transform(X)
                    X_linear = PolynomialFeatures(
                        1, include_bias=False).fit_transform(X)

                    # # linear regression from X_linear to y
                    # linear_regresion = sm.OLS(y, X_linear).fit()

                    # # polinomial regression from X_poly
                    # X_poly = PolynomialFeatures(
                    #     poly_degree, interaction_only=True, include_bias=False).fit_transform(X)
                    # polynomial_regression = sm.OLS(y, X_poly).fit()

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

        return all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm

    # set env_name and color_stats based on mode
    if mode == 0:
        env_name = f'{SYS_PATH}/functionlearning/data/claude_generated_functionlearningtasks_paramsNA_dim1_data20_tasks9991_run0_procid0_pversion2'
        color_stats = '#405A63'  # '#2F4A5A'# '#173b4f'
    # elif mode == 1:  # last plot
    #     env_name = f'{SYS_PATH}/functionlearning/data/linear_data'
    #     color_stats = '#66828F'  # 5d7684'# '#5d7684'
    elif mode == 2:  # first plot
        env_name = f'{SYS_PATH}/functionlearning/data/real_data'
        color_stats = '#173b4f'  # '#0D2C3D' #'#8b9da7'
    # elif mode == 3:
    #     env_name = f'{SYS_PATH}/functionlearning/data/synthetic_tasks_dim4_data650_tasks1000_nonlinearTrue'
    #     color_stats = '#5d7684'

    # load data
    data = pd.read_csv(f'{env_name}.csv')
    data.input = data['input'].apply(lambda x: np.array(eval(x)))
    if mode == 2:
        data.target = data['target'].apply(lambda x: np.array(eval(x)))

    if os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz'):
        stats = np.load(
            f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz', allow_pickle=True)
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear = stats['all_corr'], stats[
            'gini_coeff'], stats['posterior_logprob'], stats['all_accuraries_linear']
        all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm = stats[
            'all_accuraries_polynomial'], stats['all_targets_with_norm'], stats['all_features_with_norm']
    else:
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, \
            all_targets_with_norm, all_features_with_norm = return_data_stats(
                data)
    # gini_coeff = np.array(gini_coeff)
    # gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy()

    FONTSIZE = 22  # 8
    fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    axs[0].plot(all_accuraries_linear, color=color_stats, alpha=1., lw=3)
    # axs[0].plot(all_accuraries_polynomial, alpha=0.7)
    sns.histplot(np.array(all_features_with_norm), ax=axs[1], bins=11, binrange=(
        0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(np.stack(all_targets_with_norm).reshape(-1), ax=axs[2], bins=11, binrange=(
        0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(posterior_logprob, ax=axs[3], bins=5, binrange=(
        0.0, 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # axs[1].set_xlim(-1, 1)

    axs[0].set_ylim(0., 1.05)
    axs[1].set_ylim(0,  0.2)
    axs[2].set_ylim(0,  0.2)
    axs[2].set_xlim(0., 1.05)
    axs[3].set_xlim(0., 1.05)

    # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axs[0].set_ylabel('Squared Error', fontsize=FONTSIZE)
    axs[1].set_ylabel('Proportion', fontsize=FONTSIZE)
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    if mode == 3:
        axs[0].set_xlabel('Trials', fontsize=FONTSIZE)
        axs[1].set_xlabel('Input', fontsize=FONTSIZE)
        axs[2].set_xlabel('Target', fontsize=FONTSIZE)
        axs[3].set_xlabel('Posterior probability ', fontsize=FONTSIZE)

    # set title
    if mode == 2:
        axs[0].set_title('Performance', fontsize=FONTSIZE)
        axs[1].set_title('Input distribution', fontsize=FONTSIZE)
        axs[2].set_title('Target distribution', fontsize=FONTSIZE)
        axs[3].set_title('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_stats_' +
                str(mode) + '.svg', bbox_inches='tight')
    plt.show()

    # # save corr, gini, posterior_logprob, and all_accuraries_linear for each mode in one .npz file
    if not os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz'):
        np.savez(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz', all_corr=all_corr, gini_coeff=gini_coeff, posterior_logprob=posterior_logprob, all_accuraries_linear=all_accuraries_linear,
                 all_accuraries_polynomial=all_accuraries_polynomial, all_targets_with_norm=all_targets_with_norm, all_features_with_norm=all_features_with_norm)
