# this file will create a html map

# imports
import pandas as pd
from data_visualization import return_and_save_bike_station_map
from data_storage import connection

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

return_and_save_bike_station_map(pd.read_sql_query('''SELECT * FROM raw''', connection),
                                 pd.read_sql_query('''SELECT * FROM station_locations''', connection)).save("./images/bike_rental_stations.html")

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can now go into your parent directory and enter the folder images, there is an HTML file, which contains the map of the bike rental stations!")
