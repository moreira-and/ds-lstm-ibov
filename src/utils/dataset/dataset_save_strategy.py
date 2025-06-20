

def update_df(old_df, new_df):
    updated_df = new_df.combine_first(old_df)

    updated_df.sort_index(inplace=True)
    updated_df.ffill(inplace=True)
    updated_df.bfill(inplace=True)

    return updated_df

