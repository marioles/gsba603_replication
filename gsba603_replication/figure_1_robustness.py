import matplotlib.pyplot as plt
import pandas as pd

from . import figure_1, utils


def extract_coefficient_from_result(result):
    plot_df = figure_1.extract_values_from_result(result=result)
    coef_ss = plot_df["coefficient"]
    return coef_ss


def _format_series(dv, ss, seed):
    multi_ls = (dv, seed)
    ss.name = multi_ls
    return ss


def get_subsample_coefficient(df, seed, dependent_ls, regress_kwargs_dd):
    sample_df = df.sample(frac=0.8, random_state=seed)

    regress_kwargs_dd.update({"df": sample_df})
    result_dd = {dv: figure_1.regress_diff_in_diff(dv=dv, **regress_kwargs_dd) for dv in dependent_ls}
    coef_dd = {dv: extract_coefficient_from_result(result=result) for dv, result in result_dd.items()}
    return coef_dd


def get_stats(ls):
    df = pd.concat(ls, axis=1)
    mean_ss = df.T.mean()
    mean_ss.name = "coefficient"
    std_ss = df.T.std()
    std_ss.name = "std_error"
    stats_df = pd.concat([mean_ss, std_ss], axis=1)
    return stats_df


def generate_robustness_plot(dependent_ls, result_dd):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    iterate_ls = zip(dependent_ls, axes)
    for dv, ax in iterate_ls:
        plot_df = result_dd[dv]
        figure_1.generate_sub_plot(ax=ax, dv=dv, plot_df=plot_df)

    plt.tight_layout()
    return fig


def run_bootstrap(n_bootstrap=None):
    if n_bootstrap is None:
        n_bootstrap = 100

    read_df = utils.read_data()
    df = figure_1.pre_process_data(df=read_df)

    dependent_ls = figure_1.get_dependent_list()
    kwargs_dd = figure_1.construct_kwargs_dict(df=df)

    bootstrap_dd = {
        "df": df,
        "dependent_ls": dependent_ls,
        "regress_kwargs_dd": kwargs_dd,
    }

    seed_ls = list(range(n_bootstrap))
    coef_ls = [get_subsample_coefficient(seed=seed, **bootstrap_dd) for seed in seed_ls]
    coef_dd = {dv: [i[dv] for i in coef_ls] for dv in dependent_ls}
    stats_dd = {dv: get_stats(ls) for dv, ls in coef_dd.items()}
    panel_plot = generate_robustness_plot(dependent_ls=dependent_ls, result_dd=stats_dd)

    name = "figure_1_bootstrap.pdf"
    utils.export_plot(name=name, panel_plot=panel_plot)


def main(n_bootstrap=None):
    run_bootstrap(n_bootstrap=n_bootstrap)
