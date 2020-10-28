# this file will create a html map

# imports
import pandas as pd
from data_preprocessing import decompress_pickle
from data_visualization import return_and_save_bike_station_map

print(" ___         _           _   _\n\
(  _ \      ( )_        ( ) ( )_ \n\
| | ) |  _ _|  _)  _ _  | | | |_)____ \n\
| | | )/ _  ) |  / _  ) | | | | |_   ) \n\
| |_) | (_| | |_( (_| | | \_/ | |/ /_ \n\
(____/ \__ _)\__)\__ _)  \___/(_)____) \n\n\
            / \_/ \ \n\
            |     |  _ _ _ _\n\
            | (_) |/ _  )  _ \ \n\
            | | | | (_| | (_) ) \n\
            (_) (_)\__ _)  __/ \n\
                        | | \n\
                        (_) \n\
Creating a map with all bike rental stations in Washington D.C.!")

return_and_save_bike_station_map(pd.DataFrame(decompress_pickle("./data/interim/ArtificalRentals11.pbz2")).append(pd.DataFrame(decompress_pickle("./data/interim/ArtificalRentals12.pbz2"))),
                                 decompress_pickle("./data/interim/Stations.pbz2")).save("./images/bike_rental_stations.html")

print("Done!\n\nYou can now go into your parent directory and enter the folder images, where there is an HTML file, which contains the map of the bike rental stations!")