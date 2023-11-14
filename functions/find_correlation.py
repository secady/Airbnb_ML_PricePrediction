def func(df, value, correlation = "both", drop=True):
    '''
    Input:
    1) df pd.Dataframe
        use df.corr() of the dataframe
    2) value: float
        The threshold you want to use
    3) correlation : string
        "positive" : show only positive correlation higher than value
        "negative" : show only negative correlation lower than value
        "both" : show both
    4) drop : bool
        If you want to drop the columns with all nans (all values below/over the threshold)
    Output:
        The correlation dataframe to use in the heatmap
    '''
    if correlation=="both":
        for col in df.columns:
            df[col] = df[(df[col]>=value) | (df[col]<=-value)][col]
    elif correlation=="positive":
        for col in df.columns:
            df[col] = df[(df[col]>=value)][col]
    elif correlation=="negative":
        for col in df.columns:
            df[col] = df[(df[col]<=-value)][col]
    else:
        print("correlation must be: 'both', 'positive', 'negative'")
        return
    if drop==True:
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
    return df