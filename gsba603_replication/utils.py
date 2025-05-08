def calculate_outcomes(df):
    df["exp_nf_w"] = df["exp_nf"].sub(df["tot_hhspend"], fill_value=0)

    df["tot_exp_w"] = df["exp_nf_w"].add(df["food_w"], fill_value=0)

    df["totcons_w"] = df["exp_nf_w"].add(df["tot_hhspend"], fill_value=0)
    df["totcons_w"] = df["totcons_w"].add(df["food_w"], fill_value=0)

    df["costs_nw_w"] = df["costs_ag_w"].add(df["costs_livs_w"], fill_value=0)
    df["costs_nw_w"] = df["costs_nw_w"].add(df["costs_fs_w"], fill_value=0)
    df["costs_nw_w"] = df["costs_nw_w"].add(df["costs_nfbiz_w"], fill_value=0)

    df["REVnw_w"] = df["AgREV_w"].add(df["LREV_w"], fill_value=0)
    df["REVnw_w"] = df["REVnw_w"].add(df["FSREV_w"], fill_value=0)
    df["REVnw_w"] = df["REVnw_w"].add(df["BREV_w"], fill_value=0)
    return df


def filter_attrition(df):
    attrition_filter_ss = df["no_attrition_food"] == 1
    filter_df = df.loc[attrition_filter_ss].copy()
    return filter_df


def filter_window(df, months=24):
    window_filter_ss = df["tau"] >= -months
    window_filter_ss = window_filter_ss & (df["tau"] < months)
    window_filter_df = df.loc[window_filter_ss].copy()
    return window_filter_df
