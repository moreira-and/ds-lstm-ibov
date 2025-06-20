import pandas as pd
import holidays

def update_df(old_df, new_df):
    # Garante cópias independentes para evitar alterações externas
    old_df = old_df.copy()
    new_df = new_df.copy()

    # Substitui dados em índices já existentes (com merge por índice)
    old_df.update(new_df)

    # Adiciona as linhas novas que só existem no new_df
    is_new = ~new_df.index.isin(old_df.index)
    new_only = new_df[is_new]

    # Combina tudo
    updated_df = pd.concat([old_df, new_only])

    # Ordena e preenche se necessário
    updated_df.sort_index(inplace=True)
    updated_df.ffill(inplace=True)
    updated_df.bfill(inplace=True)

    return updated_df


def enrich_with_business_calendar(df):
    # Ensure the index is datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Day of the week as full name, e.g., "Monday"
    df['day_of_week'] = df.index.day_name()

    # Month of the year
    df['month'] = df.index.month_name()

    # Is it weekend?
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

    # Brazilian national holidays for all years in the index
    years = df.index.year.unique()
    br_holidays = holidays.Brazil(years=years)

    # Is it a national holiday?
    df['is_holiday'] = df.index.to_series().apply(lambda x: x in br_holidays)

    # Keep only business days (not weekend, not holiday)
    business_days_df = df[~df['is_weekend'] & ~df['is_holiday']]

    return business_days_df

