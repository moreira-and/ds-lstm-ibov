

def update_df(old_df, new_df):
    updated_df = new_df.combine_first(old_df)
    old_df.update(new_df)
    return updated_df

