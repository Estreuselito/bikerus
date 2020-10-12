import pandas as pd
data = pd.read_csv("BikeRental.csv")
data.to_csv("This_is_a_test.csv")
print("done man!")