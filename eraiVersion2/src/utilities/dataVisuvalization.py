def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

def displayHeatMap(correlation):
    import seaborn as sns
    import matplotlib.pyplot as plt

    cmap=sns.diverging_palette(5, 250, as_cmap=True)

    correlation.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())



    fig, ax = plt.subplots(figsize=(15,15))
    return sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)