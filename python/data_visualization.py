# This file contains all functions, which will be used for visualization    

# imports
import pandas as pd
import folium  # for creating a map
from branca.element import Template, MacroElement  # for creating the legend

def return_and_save_bike_station_map(df,
                                     station_location):
    """Function which returns a map of bike stations.

    Parameters
    ----------
    df : dataframe
        a dataframe which contains all data from the bikeshare rental company Capital Bike
    station_location : dataframe
        a dataframe which contains all Terminal IDs to bike rental stations from Capital Bike

    Returns
    -------
    map
        returns a folium html map with a draggable legend
    """
    if len(df["Start station number"].unique()) == len(df["End station number"].unique()):
        pass
    else:
        return False
    unique_stats = (pd.DataFrame({"Start station number": (df["Start station number"].unique())})
                    .merge(pd.DataFrame({"NO_OF_BIKES": (df.groupby(["Start station number"])
                                                                .count()["Bike number"])}),
                            on="Start station number", how="left")
                    .rename(columns={"Start station number": "TERMINAL_NUMBER"}))
    unique_stats["color"] = pd.cut(unique_stats['NO_OF_BIKES'],
                                bins=3,
                                labels=['green', 'orange', 'red'])
    cuts = pd.cut(unique_stats['NO_OF_BIKES'],
                    bins=3,
                    retbins=True,
                    labels=['green', 'orange', 'red'])
    station_loc_full = pd.merge(unique_stats, station_location[["TERMINAL_NUMBER", "LONGITUDE", "LATITUDE", "ADDRESS"]], on = "TERMINAL_NUMBER")
    m = folium.Map(location=[((station_loc_full.LATITUDE.min() + station_loc_full.LATITUDE.max())/2), ((station_loc_full.LONGITUDE.min() +  station_loc_full.LONGITUDE.max()) / 2)],
                zoom_start = 12,
                no_touch=True,
                control_scale=True)
                

    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>jQuery UI Draggable - Default functionality</title>
        <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

        <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
        
        <script>
        $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

        </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
        
    <div class='legend-title'>Legend of usage over two years</div>
    <div class='legend-scale'>
        <ul class='legend-labels'>
        <li><span style='background:red;opacity:0.7;'></span> High usage (""" + str(round(cuts[1][2])) + """ to """ + str(round(cuts[1][3])) + """)</li>
        <li><span style='background:orange;opacity:0.7;'></span>Medium usage (""" + str(round(cuts[1][1])) + """ to """ + str(round(cuts[1][2])) + """)</li>
        <li><span style='background:green;opacity:0.7;'></span>Small usage (0 to """ + str(round(cuts[1][1])) + """)</li>

        </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
        .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
        .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
        .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
        .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
        .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
        .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template)

    for i in range(0, len(station_loc_full)-1):
        folium.Marker([station_loc_full["LATITUDE"][i], station_loc_full["LONGITUDE"][i]], icon=folium.Icon(color=station_loc_full["color"][i], icon = "bicycle", prefix='fa')).add_to(m)

    m = m.get_root().add_child(macro)

    return m