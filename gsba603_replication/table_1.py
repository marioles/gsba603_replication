import pandas as pd
import statsmodels.formula.api as smf

from . import utils


def generate_post_treatment(df, tau):
    df["post"] = df[tau].apply(lambda x: 1 if x >= 0 else 0)
    return df


def pre_process_data(df, tau="tau"):
    # Two-year analysis window
    window_filter_df = utils.filter_window(df=df, months=24)

    # No-attrition filter
    filter_df = utils.filter_attrition(df=window_filter_df)

    # Calculations
    filter_df = utils.calculate_outcomes(df=filter_df)

    # Generate treatment interaction
    filter_df = generate_post_treatment(df=filter_df, tau=tau)

    return filter_df


def regress_diff_in_diff(df, dv, tau, treatment, post, fe=None, control=None, clustvar=None):
    copy_df = df.copy()

    if treatment == "Treatment":
        rename_dd = {"Treatment": "treatment"}
        copy_df = copy_df.rename(columns=rename_dd)
        treatment = "treatment"

    iter_ls = [
        dv,
        tau,
        treatment,
        post,
        clustvar,
        fe,
        control,
    ]
    keep_ls = list()
    for i in iter_ls:
        if isinstance(i, list):
            keep_ls += i
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
        copy_df[clustvar] = pd.Categorical(copy_df[clustvar], ordered=True)

    # Create formula
    formula = f"{dv} ~ {post}:{treatment} + {treatment} + {post}"
    if isinstance(fe, list):
        fe = fe.copy()
        if tau in fe:
            fe.remove(tau)
            fe.append(tau_cat)
        fe_formula_ls = [f"C({i})" for i in fe]
        fe_formula = " + ".join(fe_formula_ls)
    else:
        if fe == tau:
            fe = tau_cat
        fe_formula = f"C({fe})"
    formula = f"{formula} + {fe_formula}"

    if control is not None:
        if isinstance(control, list):
            control_formula = " + ".join(control)
        else:
            control_formula = control
        formula = f"{formula} + {control_formula}"

    model = smf.ols(formula, data=copy_df)
    if clustvar is not None:
        cluster_ss = copy_df[clustvar].copy()
        cluster_dd = {"groups": cluster_ss}
        results = model.fit(cov_type="cluster", cov_kwds=cluster_dd)
    else:
        results = model.fit()

    return results


def get_post_treat(post, treatment):
    if treatment == "Treatment":
        treatment = "treatment"

    post_treat = f"{post}:{treatment}"
    return post_treat


def generate_column(result, df, dv, treatment, post, tau):
    post_treat = get_post_treat(post=post, treatment=treatment)
    coefficient = result.params[post_treat]
    std_error = result.bse[post_treat]

    baseline = df[dv].mean()
    observations = df.shape[0]

    n_events = ((df[treatment] == 1) & (df[tau] == 0)).sum()
    adj_r_squared = result.rsquared_adj

    dd = {
        "coefficient": coefficient,
        "std_error": std_error,
        "baseline": baseline,
        "observations": observations,
        "n_events": n_events,
        "adj_r_squared": adj_r_squared,
    }

    ss = pd.Series(dd, name=dv)
    return ss


def get_rename_col_dict():
    dd = {
        "a_symptom": "Reported symptoms",
        "tot_hhspend": "Health spending",
        "tot_exp_w": "Nonhealth spending",
        "totcons_w": "Total spending",
        "costs_nw_w": "Business spending",
        "hours_hired": "Hired labor",
        "hours_hhlab": "Household labor",
        "REVnw_w": "Revenues",
    }
    return dd


def get_rename_row_dict():
    dd = {
        "coefficient": "Post x Treatment",
        "std_error": "",
        "baseline": "Baseline mean (DV)",
        "observations": "Observations",
        "n_events": "Number of events",
        "adj_r_squared": "Adj R2",
    }
    return dd


def get_regression_result(df, dv, tau, treatment, post, fe=None, control=None, clustvar=None):
    result = regress_diff_in_diff(df=df,
                                  dv=dv,
                                  tau=tau,
                                  treatment=treatment,
                                  post=post,
                                  fe=fe,
                                  control=control,
                                  clustvar=clustvar)
    col_ss = generate_column(result=result, df=df, dv=dv, treatment=treatment, post=post, tau=tau)
    return col_ss


def get_panel_text(panel):
    if panel == "a":
        text = "Panel A. Using shocks occurring during the first half of the sample"
    else:
        text = "Panel B. Using all shocks"
    return text


def format_table(column_ls, panel):
    df = pd.concat(column_ls, axis=1)
    rename_col_dd = get_rename_col_dict()
    rename_row_dd = get_rename_row_dict()
    rename_df = df.rename(columns=rename_col_dd, index=rename_row_dd)

    index_ls = rename_df.index
    panel_text = get_panel_text(panel=panel)
    multi_ls = [(panel_text, i) for i in index_ls]
    multi_idx_ls = pd.MultiIndex.from_tuples(multi_ls)

    rename_df.index = multi_idx_ls
    return rename_df


def export_table(name, table_df):
    export_path = utils.get_export_path()
    path = f"{export_path}/{name}"
    table_df.to_csv(path)


def generate_table_1():
    read_df = utils.read_data()
    df = pre_process_data(df=read_df)

    dependent_ls = [
        "a_symptom",
        "tot_hhspend",
        "tot_exp_w",
        "totcons_w",
        "costs_nw_w",
        "hours_hired",
        "hours_hhlab",
        "REVnw_w",
    ]

    kwargs_dd = {
        "tau": "tau",
        "treatment": "Treatment",
        "post": "post",
        "clustvar": "id",
        "fe": ["id", "month", "tau"],
        "control": ["Nm", "Nf", "headage", "mean_edu"],
    }

    # Panel A
    first_half_df = utils.filter_first_half_shock(df=df)
    column_a_ls = [get_regression_result(dv=dv, df=first_half_df, **kwargs_dd) for dv in dependent_ls]
    panel_a_df = format_table(column_ls=column_a_ls, panel="a")

    # Panel B
    column_b_ls = [get_regression_result(dv=dv, df=df, **kwargs_dd) for dv in dependent_ls]
    panel_b_df = format_table(column_ls=column_b_ls, panel="b")

    concat_ls = [panel_a_df, panel_b_df]
    table_df = pd.concat(concat_ls, axis=0)

    name = "table_1.csv"
    export_table(name=name, table_df=table_df)


def main():
    generate_table_1()
