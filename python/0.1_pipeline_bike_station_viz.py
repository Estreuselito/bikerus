# this file will create a html map

# imports
import pandas as pd
from data_preprocessing import decompress_pickle
from data_visualization import return_and_save_bike_station_map

return_and_save_bike_station_map(pd.DataFrame(decompress_pickle("./data/interim/ArtificalRentals11.pbz2")).append(pd.DataFrame(decompress_pickle("./data/interim/ArtificalRentals12.pbz2"))),
                                 decompress_pickle("./data/interim/Stations.pbz2")).save("./images/bike_rental_stations.html")