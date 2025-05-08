import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.formula.api as smf

from . import utils


def _compute_distance(x):
    if not pd.isna(x):
        if x != -1 and x != 0:
            close_tot = 1 / x
        else:
            close_tot = 0
    else:
        close_tot = 0
    return close_tot


def pre_process_data(df):
    # Drop attritors
    filter_ss = df["no_attrition_food"] == 1
    filter_ss = filter_ss & df["no_attrition_food_j"] == 1
    attritors_df = df.loc[filter_ss].copy()

    # Post_i indicator
    post_tau_i_ss = (attritors_df["tau_i"] >= 0) & (attritors_df["tau_i"] < 12)
    post_tau_ss = (attritors_df["tau"] > -12) & (attritors_df["tau"] < 12)
    post_ss = post_tau_i_ss & post_tau_ss
    post_ss = post_ss.astype(int)
    attritors_df["post_i"] = post_ss

    # Compute shocks_i
    groupby_ls = ["id", "id_j"]
    attritors_df["shocks_i"] = attritors_df.groupby(groupby_ls)["post_i"].transform("sum")
    attritors_df["shocks_i"] = attritors_df["shocks_i"].apply(lambda x: 1 if x > 0 else x)

    # Recode tau
    recode_df = utils.recode_tau(df=attritors_df)

    recode_df["post"] = (recode_df["tau"] > 0).astype(int)
    recode_df["c_tot"] = recode_df["Tot_geo"].isin([0, 1])
    recode_df["c_tot"] = recode_df["Tot_geo"].apply(lambda x: np.nan if pd.isna(x) else x)

    # Distance
    recode_df["close_tot"] = recode_df["Tot_geo"].apply(_compute_distance)

    # Other dummies and variables
    recode_df["anyHLABOUT"] = (recode_df["HLABOUT"] > 0).astype(int)
    recode_df["anyHLABOUT"] = recode_df.apply(lambda x: np.nan if pd.isna(x["HLABOUT"]) else x["anyHLABOUT"], axis=1)

    recode_df["anyOUTPUTOUT"] = (recode_df["OUTPUTOUT"] > 0).astype(int)
    recode_df["anyOUTPUTOUT"] = recode_df.apply(lambda x: np.nan if pd.isna(x["OUTPUTOUT"]) else x["anyOUTPUTOUT"],
                                                axis=1)

    recode_df["OUTPUTOUT"] = recode_df["OUTPUTOUT"].add(recode_df["INPUTOUT"], fill_value=0)
    recode_df["OUTPUTIN"] = recode_df["OUTPUTIN"].add(recode_df["INPUTIN"], fill_value=0)

    recode_df["OUTPUT"] = recode_df["OUTPUTOUT"].add(recode_df["OUTPUTIN"])
    recode_df["HLAB"] = recode_df["HLABOUT"].add(recode_df["HLABIN"])
    recode_df["transactions"] = recode_df["OUTPUT"].add(recode_df["HLAB"])

    # Filter shocks
    shock_filter_ss = recode_df["_degree_Tot_t_j"] > 0
    shock_filter_ss = shock_filter_ss & (~recode_df["_degree_Tot_t_j"].isna())
    keep_df = recode_df.loc[shock_filter_ss].copy()

    indirect_filter_ss = keep_df["shocks_i"] == 0
    keep_df = keep_df.loc[indirect_filter_ss].copy()
    return keep_df


def regress_diff_in_diff(df, dv, tau, h, fe=None, fe_inter=None, control=None, clustvar=None, cluster=False):
    copy_df = df.copy()

    iter_ls = [
        dv,
        tau,
        h,
        clustvar,
        fe,
        fe_inter,
        control,
    ]
    keep_ls = list()
    for i in iter_ls:
        if isinstance(i, list):
            for j in i:
                if isinstance(j, tuple):
                    k, l = j
                    keep_ls.append(k)
                    keep_ls.append(l)
                else:
                    keep_ls.append(j)
        else:
            keep_ls.append(i)
    keep_ls = list(set(keep_ls))
    copy_df = copy_df.loc[:, keep_ls].copy()
    copy_df = copy_df.dropna()

    # Handle categorical variable
    tau_ss = copy_df[tau].copy()
    category_ls = sorted(tau_ss.dropna().unique())
    tau_cat_ss = pd.Categorical(tau_ss, categories=category_ls, ordered=True)
    tau_cat = f"{tau}_cat"
    copy_df[tau_cat] = tau_cat_ss

    if clustvar is not None:
        if isinstance(clustvar, list):
            for clustvar_i in clustvar:
                copy_df[clustvar_i] = pd.Categorical(copy_df[clustvar_i], ordered=True)
        else:
            copy_df[clustvar] = pd.Categorical(copy_df[clustvar], ordered=True)

    # Create formula
    formula = f"{dv} ~ C({tau_cat}, Treatment(reference=-1)):{h}"
    formula = f"{formula} + C({tau_cat}, Treatment(reference=-1)) + {h}"
    if fe is not None:
        if isinstance(fe, list):
            fe_formula_ls = [f"C({i})" for i in fe]
            fe_formula = " + ".join(fe_formula_ls)
        else:
            fe_formula = f"C({fe})"
        formula = f"{formula} + {fe_formula}"

    if fe_inter is not None:
        # Assumed that i is continuous and j dummy
        fe_inter_formula_ls = [f"{i}:C({j})" for i, j in fe_inter]
        fe_inter_formula = " + ".join(fe_inter_formula_ls)
        formula = f"{formula} + {fe_inter_formula}"

    if control is not None:
        if isinstance(control, list):
            control_formula = " + ".join(control)
        else:
            control_formula = control
        formula = f"{formula} + {control_formula}"

    model = smf.ols(formula, data=copy_df)
    results = model.fit()

    if cluster:
        cluster_1, cluster_2 = clustvar
        cluster_1_ss = copy_df[cluster_1].copy()
        cluster_2_ss = copy_df[cluster_2].copy()
        covariance = utils.cov_cluster_2way(model=results, cluster1=cluster_1_ss, cluster2=cluster_2_ss)
        results.clustered_bse = np.sqrt(np.diag(covariance))

    return results


def get_regex():
    regex_str = r"^C\(tau_cat, Treatment\(reference=-1\)\)\[T\.(.*?)\]:close_tot$"
    return regex_str


def get_title(dv):
    title_dd = {
        "transactions": "Panel A. Total transactions\nTransactions",
        "OUTPUT": "Panel B. Supply-chain (sales) network transactions\nNumber of transactions (output/input)",
        "HLAB": "Panel C. Labor network transactions\nNumber of transactions (labor markets)",
        "tincome_w": "Panel D. Total income\nNet income",
        "exp_w": "Panel E. Consumption spending\nConsumption spending",
    }
    title = title_dd[dv]
    return title


def get_x_title(dv):
    x_title = "Time to event (half year)"
    return x_title


def get_y_title(dv):
    y_title_dd = {
        "transactions": "Number links",
        "OUTPUT": "Number links",
        "HLAB": "Number links",
        "tincome_w": "THB",
        "exp_w": "THB",
    }
    y_title = y_title_dd[dv]
    return y_title


def plot_from_data(ax, plot_df, dv, confidence_ls):
    title = get_title(dv=dv)
    x_title = get_x_title(dv=dv)
    y_title = get_y_title(dv=dv)

    x_ss = plot_df.index
    y_ss = plot_df["coefficient"]

    ax.plot(x_ss, y_ss, marker="o", linewidth=2, label="Coef")

    for confidence in confidence_ls:
        lower_label = f"lower_{int(100 * confidence)}"
        upper_label = f"upper_{int(100 * confidence)}"
        lower_ss = plot_df[lower_label]
        upper_ss = plot_df[upper_label]
        plot_label = f"{int(100 * confidence)}% CI"
        plot_color = utils.get_confidence_color(confidence)
        ax.fill_between(x_ss, lower_ss, upper_ss, color=plot_color, alpha=0.25, label=plot_label)

    ax.axhline(0, color="maroon", linestyle="--")
    ax.axvline(-1, color="maroon", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend()
    ax.grid(True)


def generate_sub_plot(ax, dv, result):
    regex_str = get_regex()

    raw_dd = {
        "coefficient": result.params,
        "std_error": result.bse,
    }

    extract_dd = {k: utils.extract_relevant_values(ss=ss, regex_str=regex_str) for k, ss in raw_dd.items()}
    concat_dd = {k: utils.append_baseline(ss=ss) for k, ss in extract_dd.items()}
    plot_df = pd.DataFrame(concat_dd)

    confidence_ls = utils.get_confidence_list()
    for confidence in confidence_ls:
        interval = (1 - confidence) / 2 + confidence
        critical_value = sp.stats.norm.ppf(interval)
        lower_label = f"lower_{int(100 * confidence)}"
        upper_label = f"upper_{int(100 * confidence)}"
        plot_df[lower_label] = plot_df["coefficient"] - critical_value * plot_df["std_error"]
        plot_df[upper_label] = plot_df["coefficient"] + critical_value * plot_df["std_error"]

    plot_from_data(ax=ax, plot_df=plot_df, dv=dv, confidence_ls=confidence_ls)


def generate_plot(dependent_ls, result_dd):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    iterate_ls = zip(dependent_ls, axes[-1])
    for dv, ax in iterate_ls:
        result = result_dd[dv]
        generate_sub_plot(ax=ax, dv=dv, result=result)

    plt.tight_layout()
    return fig


def generate_figure_2(cluster=None):
    if cluster is None:
        cluster = False

    read_df = utils.read_data(file="dyads_es_max")
    df = pre_process_data(df=read_df)

    dependent_ls = [
        "transactions",
        "OUTPUT",
        "HLAB",
        "tincome_w",
        "exp_w",
    ]

    kwargs_dd = {
        "df": df,
        "tau": "tau",
        "h": "close_tot",
        "clustvar": ["id", "id_j"],
        "fe": ["id", "month"],
        "fe_inter": [("_degree_Tot_t", "month")],
        "control": ["Nm", "Nf", "headage", "mean_edu"],
        "cluster": cluster,
    }

    result_dd = {dv: regress_diff_in_diff(dv=dv, **kwargs_dd) for dv in dependent_ls}
    panel_plot = generate_plot(dependent_ls=dependent_ls, result_dd=result_dd)

    name = "figure_2.pdf"
    utils.export_plot(name=name, panel_plot=panel_plot)


def main(cluster=None):
    generate_figure_2(cluster=cluster)
