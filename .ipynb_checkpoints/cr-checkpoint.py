import numpy as np 
import pandas as pd 
from warnings import filterwarnings
filterwarnings('ignore')
import altair as alt

df = pd.read_csv('/Users/ishan-college/Downloads/clash_royale_cards.csv')
df['Previous Win Rate'] = (df['Win Rate'] / (100 + df['Win Rate Change']) * 100).round(2)
df['Previous Usage'] = (df['Usage'] / (100 + df['Usage Change']) * 100).round(2)
df.head(2)

nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Card'], empty='none')

main = alt.Chart(df, title = 'Cards with low win rate previously get chosen less now').mark_point().encode(
    alt.X('Previous Win Rate:Q', scale = alt.Scale(zero = False), axis = alt.Axis(grid = False), title = 'Previous Win Rate'),
    alt.Y('Usage Change:Q', scale = alt.Scale(zero = False), axis = alt.Axis(grid = False, format = '%'), title = 'Usage Change'),
    alt.Tooltip(['Card']),
    color=alt.condition(nearest, alt.value('blue'), alt.value('lightgray')),

).add_selection(
    nearest
)

red_points = alt.Chart(df).transform_filter(
    '(datum["Previous Win Rate"] < 50) && (datum["Usage Change"] < 0)'
).mark_point().encode(
    alt.X('Previous Win Rate:Q'),
    alt.Y('Usage Change:Q'),
    color=alt.condition(nearest, alt.value('blue'), alt.value('red')),
)

hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#e6e6e6').encode(y='y')
vline = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(color='#e6e6e6').encode(x='x')

hline + vline + main + red_points

nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Card'], empty='none')

main = alt.Chart(df, title = 'Cards with >50% win rate previously did not show a pattern').mark_point().encode(
    alt.X('Previous Win Rate:Q', scale = alt.Scale(zero = False), axis = alt.Axis(grid = False), title = 'Previous Win Rate'),
    alt.Y('Usage Change:Q', scale = alt.Scale(zero = False), axis = alt.Axis(grid = False, format = '%'), title = 'Usage Change'),
    alt.Tooltip(['Card']),
    color=alt.condition(nearest, alt.value('blue'), alt.value('lightgray')),

).add_selection(
    nearest
)

green_points = alt.Chart(df).transform_filter(
    '(datum["Previous Win Rate"] > 50)'
).mark_point().encode(
    alt.X('Previous Win Rate:Q'),
    alt.Y('Usage Change:Q'),
    color=alt.condition(nearest, alt.value('blue'), alt.value('forestgreen')),
)

hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#e6e6e6').encode(y='y')
vline = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(color='#e6e6e6').encode(x='x')

hline + vline + main + green_points

