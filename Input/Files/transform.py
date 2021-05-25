import pandas as pd

data_x = pd.read_csv("D:/Users/F.Moraglio/Documents/CER/RECOpt/Input/Files/Profiles/consumption_profiles_month_we.csv",
			sep = ";",
			decimal= ",")
#%%
data_x.to_csv("D:/Users/F.Moraglio/Documents/CER/RECOpt/Input/Files/Profiles/consumption_profiles_month_we.csv",
		   sep = ";",
		   decimal = ".")