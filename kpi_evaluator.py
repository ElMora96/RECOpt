from numpy_financial import npv, irr
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def cash_flows(y_returns, pv_size, maintenance = 12.5 , beta = 1):
	"""Compute CF - Cash Flow given iterable of returns
	Parameters:
	y_returns: Iterable with yearly returns
	pv_size: float. Max power of PV system (KW)
	maintenance: float. Specific cost of PV
	beta: float in (0,1) -- Producer Benefit Share
	Returns:
	CF: array with yearly cash flows (or float if single value is passed)
	"""
	CF = []
	if not hasattr(y_returns, "iter"):
		y_returns = [y_returns]
	for amount in y_returns:
			cashflow = amount*beta - maintenance*pv_size
			CF.append(cashflow)
	if len(CF) == 1:
		return CF[0]
	else:
		return np.array(CF)

def yearly_returns(pv_production, zonal_prices, shared_energy, TP = 110, CU_af = 8.5):
	"""Compute (Approximate) yearly returns given yearly series.
	Parameters:
	pv_production: array_like -- yearly series photovoltaic production
	zonal_prices: array_like -- yearly series zonal electric prices 
	shared_energy: pd.Series with datetime index -- yearly series shared energy in REC
	TP: float -- Tariff Premium in REC
	CU_af: float or iterable of length 12 -- Corrispettivo Unitario di autoconsumo forfettario
	"""
	#Error check
	if not (len(pv_production) == len(zonal_prices) == len(shared_energy)):
			raise ValueError("Series must be equal length")

	#Compute Returns
	returns_pv = np.multiply(pv_production, zonal_prices) #PV sales
	returns_se = np.multiply(shared_energy, TP - zonal_prices) #Contributions
	monthly_se = shared_energy.resample("M").sum().values #Monthly shared energy volumes
	if not hasattr(CU_af, "iter"):
		CU_af = [CU_af] * 12
	if len(CU_af) != 12:
		raise ValueError("Please provide monthy values for CU_af")
	returns_rb = np.multiply(monthly_se, CU_af) #Reimbursments

	return np.sum(returns_pv) + np.sum(returns_se) + np.sum(returns_rb)

def NPV(cflow, discount, investment):
	"""Compute NPV - Net Present Value.
	Parameters:
	cflow: array_like -- Contains yearly cash flows, excluding initial investment
	investment: (positive) float -- Initial investment
	discount: float in [0,1] -- Investment Discount Rate
	Returns float NPV.
	"""
	cflow = list(cflow) #cast as list
	cflow = [-investment] + cflow #Add (negative) initial investment to cash flows vector
	result = npv(discount, cflow)
	return result

def IRR(cflow, investment):
	"""Compute IRR - Internal Rate of Return
	Parameters:
	cflow: array_like -- Contains yearly cash flows, excluding initial investment
	investment: (negative) float -- Initial investment
	Returns float IRR.
	"""
	cflow = list(cflow) #cast as list
	cflow = [-investment] + cflow #Add (negative) initial investment to cash flows vector
	result = irr(cflow)
	return result

def build_pv_profile(timerange, profile_pv, power_max = 45):
	"""Construct hourly PV production profile given average power profiles
	timerange: pd.date_range -- build profile over this range
	profile_pv: array_like of shape (24,12). -- Average PV profiles.
	"""
	vals = []
	for tx in timerange:
		hh = tx.hour 	
		mm = tx.month - 1
		val = profile_pv[hh, mm]  * power_max * (1/1000)
		vals.append(val)
	result = pd.Series(vals, index = timerange)
	return result

def build_shared_energy_profile(timerange, profile_wd, profile_we):
	"""Construct hourly shared energy profile given average power profiles.
	Parameters:
	timerange: pd.date_range -- build profile over this range
	profile_wd: array_like of shape (24,12). SP profile for weekdays
	profile_we: array_like of shape (24,12). SP profile for weekend days. 
	
	Returns:
	result: pd.Series with timerange index.
	"""
	profiles = {0: profile_wd,
				1: profile_we}
	vals = [] #store values
	for tx in timerange:
		hh = tx.hour 
		dd = tx.dayofweek
		mm = tx.month - 1 #consider months in [0, ..., 11]
		daytype = 0 if dd < 5 else 1 #weekday or weekend
		prof = profiles[daytype] #correct profile
		val = prof[hh, mm]*(1/1000)#Since power is hourly, no transformation is needes
		vals.append(val)
	result = pd.Series(vals, index = timerange)
	return result

#%%
#UNIT TEST#

if __name__ == '__main__':
	profile_wd = pd.read_csv("D:/Users/F.Moraglio/Documents/CER/RECOpt/Output/Files/Shared_Power/shared_power_pv_45.0_battery_10.0/wd.csv",
							 sep = ";", decimal = ",").values
	profile_we = pd.read_csv("D:/Users/F.Moraglio/Documents/CER/RECOpt/Output/Files/Shared_Power/shared_power_pv_45.0_battery_10.0/we.csv",
							 sep = ";", decimal = ",").values	
	prices = pd.read_excel("D:/Users/F.Moraglio/Documents/CER/RECOpt/Input/Files/Prices/2019.xlsx",
						engine='openpyxl',
						sheet_name = 0,
						index_col = 0).values.ravel()
	
	pv = pd.read_csv("D:/Users/F.Moraglio/Documents/CER/RECOpt/Input/pv_production_unit.csv",
					sep = ";",
					decimal = ".")
	pv = pv.values[:,1:]
	time = pd.date_range(start = "2019-01-01",
					     end ="2019-12-31 23:00",
						 freq = "H",
						 closed = None
						 )
	
	shared_energy_sample = build_shared_energy_profile(time, profile_wd, profile_we)
	pv_sample = build_pv_profile(time, pv, 45)
	shared_energy_sample.plot()
	plt.show()
	#Compute sample quantities
	y_returns_sample = yearly_returns(pv_sample, prices, shared_energy_sample )
	y_returns_sample = cash_flows(y_returns_sample, 45, beta = 0.5)
	#Cash flows ab minchia
	cf =  [y_returns_sample] * 20
	irr = IRR(cf, 500*10 + 45*810)
	
	
	