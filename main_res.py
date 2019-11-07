import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


if __name__ == "__main__":
    # fname = "./Applebees-USA.csv"
    fname = "./Mcdonalds_USA_CAN.csv"
    df = pd.read_csv(
        fname, header=None, sep=",", names=["longitude", "latitude", "name", "address"]
    )
    df['state'] = df['name'].apply(lambda x: x.split(',')[-1])
    val = df['state'].value_counts()
    # val.plot.bar()
    # plt.show()

    # plot in state heatmap
    fig = go.Figure(data=go.Choropleth(
        locations=val.index.tolist(), # Spatial coordinates
        z = val.values.astype(np.float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "# of Applebee's",
    ))
    fig.update_layout(
        title_text = "Number of Applebee\'s in US",
        geo_scope='usa', # limite map scope to USA
    )

    fig.write_html("state_heatmap.html", auto_open=True)


