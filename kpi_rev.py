from numpy_financial import npv, irr
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

def initial_investment(pv_size, battery_size, n_batteries = 1, capex_pv = 810, capex_batt = 509):
	"""Compute initial investment"""
	return pv_size*capex_pv + battery_size*capex_batt*n_batteries

def cash_flows(fed_energy, shared_energy, pv_size, beta, PR3 = 42,  CUAF = 8.56 , TP = 110, inf_rate = 0.02, OM = 12.5, n_years = 20):
	"""Compute cash flows over 20 years"""
	prezzi_ritiro = [PR3]*n_years
	for i in range(1, n_years):
		prezzi_ritiro[i] = (1 + inf_rate)*prezzi_ritiro[i-1] #prezzi minimi garantiti inflazionati
 	#Assumo CUAF costante nei prox 20 anni
	gse_refund = np.array([shared_energy * CUAF] * n_years)
	#TP is guaranteed constant
	premium = np.array([shared_energy * TP] * n_years)
	contributions = gse_refund + premium #Amount to be splitted
	contrib_prod = beta * contributions #Goes to producer
	contrib_other = (1 - beta) * contributions #Goes to non producer
	#Sales
	energy_sales = np.array([fed_energy * prezzo for prezzo in prezzi_ritiro])
	#Cash cash_flows (producer)
	cflows = np.array([rce + contr - OM*pv_size for rce, contr in zip(energy_sales, contrib_prod)])

	return cflows, energy_sales, contributions, contrib_prod   


#%%
#UNIT TEST#
if __name__ == '__main__':
	pass