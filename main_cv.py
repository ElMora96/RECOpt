import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import sys
# The basepath of the file is stored in a variable 
basepath = Path(__file__).parent
sys.path.append(basepath.as_posix())
import pandas as pd
#REDUCED MAIN TO PERFORM CROSS VALIDATION

from scipy.interpolate import interp1d
from tabulate import tabulate
import parameters_input as inp

import plot_generator as plot
import datareader
from tictoc import tic, toc
from shared_energy_evaluator import shared_energy_evaluator
from kpi_rev import *

### Parameters setup
power_max = 140.5#105.5 - 140.5
dt_aggr = 60 

#NB: SET PARAMETRIC
# Photovoltaic (PV)
pv_setup, pv_size_range = inp.simulation_setup('PV')

# Battery 
battery_setup, battery_size_range = inp.simulation_setup('battery')

#Do not show setup
#Misc Dictionaries 
seasons = {
	'winter': (0, 'w'),
	'spring': (1, 'ap'),
	'summer': (2, 's'),
	'autumn': (3, 'ap')
	}

days = {
	'week-day': (0, 'wd'),
	'weekend-day': (1, 'we')
	}

months = {
	'january': {'id': (0, 'jan'), 'season': 'winter', 'days_distr': {'week-day': 23, 'weekend-day': 8}},
	'february': {'id': (1, 'feb'), 'season': 'winter', 'days_distr': {'week-day': 20, 'weekend-day': 8}},
	'march': {'id': (2, 'mar'), 'season': 'winter', 'days_distr': {'week-day': 22, 'weekend-day': 9}},
	'april': {'id': (3, 'apr'), 'season': 'spring', 'days_distr': {'week-day': 21, 'weekend-day': 9}},
	'may': {'id': (4, 'may'), 'season': 'spring', 'days_distr': {'week-day': 23, 'weekend-day': 8}},
	'june': {'id': (5, 'jun'), 'season': 'spring', 'days_distr': {'week-day': 21, 'weekend-day': 9}},
	'july': {'id': (6, 'jul'), 'season': 'summer', 'days_distr': {'week-day': 22, 'weekend-day': 9}},
	'august': {'id': (7, 'aug'), 'season': 'summer', 'days_distr': {'week-day': 23, 'weekend-day': 8}},
	'september': {'id': (8, 'sep'), 'season': 'summer', 'days_distr': {'week-day': 20, 'weekend-day': 10}},
	'october': {'id': (9, 'oct'), 'season': 'autumn', 'days_distr': {'week-day': 23, 'weekend-day': 8}},
	'november': {'id': (10, 'nov'), 'season': 'autumn', 'days_distr': {'week-day': 22, 'weekend-day': 8}},
	'december': {'id': (11, 'dec'), 'season': 'autumn', 'days_distr': {'week-day': 21, 'weekend-day': 10}},
	}	

# The days distribution in the seasons can be evaluated as well
days_distr = {}
for month in months:
		season = months[month]['season']
		
		if season not in days_distr: days_distr[season] = {'week-day': months[month]['days_distr']['week-day'],
														   'weekend-day': months[month]['days_distr']['weekend-day']}

		else: 
				days_distr[season]['week-day'] += months[month]['days_distr']['week-day']
				days_distr[season]['weekend-day'] += months[month]['days_distr']['weekend-day']

# Storing some useful quantities
n_months = len(months)
n_seasons = len(seasons)
n_days = len(days)
	
auxiliary_dict = {
	'seasons': seasons,
	'n_seasons': n_seasons,
	'months': months,
	'n_months': n_months,
	'days': days,
	'n_days': n_days,
	'days_distr': days_distr,
	}

# Here the core of the simulation starts
tic()	
#Time variables
time = 24
dt = 1 #dt_aggr/60
time_sim = np.arange(0, time, dt)
time_length = np.size(time_sim)
time_dict = {
	'time': time,
	'dt': dt,
	'time_sim': time_sim,
	'time_length': time_length,
	}

# Maximum power from the grid (total) (kW)
grid_power_max = power_max
battery_specs = datareader.read_param('battery_specs.csv', ';', 'Input')

# Storing the information about the various technologies considered in a dictionary
# that is passed to the various methods
technologies_dict = {
	'grid_power_max': grid_power_max,
	'battery_specs': battery_specs,
	}

#PV
data_pv = datareader.read_general('pv_production_unit.csv', ';', 'Input')
time_pv = data_pv[:, 0]
pv_production_unit = data_pv[:, 1:]

# Checking if there is already a file where the load profiles for this configuration have been stored
dirname = Path('Input')
subdirname = Path('Files')
subsubdirname = Path('Profiles')

# If the files exist, program reads it
try:
	#data_wd = datareader.read_general('consumption_profiles_month_wd.csv', ';', '/'.join((dirname, subdirname, subsubdirname)))
	data_wd = pd.read_csv(basepath / dirname / subdirname / subsubdirname / 'consumption_profiles_month_wd.csv',
						  sep = ";",
						  decimal = ",",
						  ).dropna().values
	data_we = pd.read_csv(basepath / dirname / subdirname / subsubdirname / 'consumption_profiles_month_we.csv',
						  sep = ";",
						  decimal = ",",
						  ).dropna().values
	consumption_month_wd = data_wd[:, 1:]
	consumption_month_we = data_we[:, 1:]
	consumption_month_day = np.stack((consumption_month_wd, consumption_month_we), axis = 2)
except:
	raise RuntimeError("Must provide load profiles in proper format.") # load_profiler_flag = 1

	# Total number of configurations to be analysed
n_pv_sizes = len(pv_size_range)
n_battery_sizes = len(battery_size_range)
n_configurations = n_pv_sizes*n_battery_sizes

fixed_analysis_flag = 0
if n_configurations == 1: 
	#fixed_analysis_flag = 1
	raise RuntimeError("Please Set Parametric Analysis")

# Initializing a list where to store some relevant results to be printed as a tabulate
tab_results = []

# Initializing arrays where to store some relevant results to be plotted in case of
# parametric analysis
if n_configurations  != 1:

	# Shared energy in a year for each configuration
	esh_configurations = np.zeros((n_pv_sizes, n_battery_sizes))

	# Self-sufficiency index in a year for each configuration
	ssi_configurations = np.zeros((n_pv_sizes, n_battery_sizes))

	# Self-consumption index in a year for each configuration
	sci_configurations = np.zeros((n_pv_sizes, n_battery_sizes))

	# A message is printed to let the user know how the simulation is proceeding (every 25% of progress)
perc_configurations = 25
count_configurations = 0

# The user is informed that the evaluation of the shared energy is starting
message = '\nEvaluation of the energy shared by the aggregate of PODs for {:d} configuration(s).'.format(n_configurations)
print(message)

# Creating the input power dictionary to be passed to the method  that evaluates the shared energy
# for the configuration
input_powers_dict = {
	'pv_production_unit': pv_production_unit,
	'consumption_month_day': consumption_month_day,
	}

# Running through the different sizes for the PV
tic()
for pv_size in pv_size_range:

	# Storing the index of the PV size in the pv_size_range list (needed for properly storing the results)
	pv_index = pv_size_range.index(pv_size)

	# Storing the PV size in the technologies_dict that is passed to the methods
	technologies_dict['pv_size'] = pv_size

	# Running through the different sizes for the battery
	for battery_size in battery_size_range:

		# Storing the index of the battery size in the battery_size_range list (needed for properly storing the results)
		battery_index = battery_size_range.index(battery_size)

		# Storing the battery size in the technologies_dict that is passed to the methods
		technologies_dict['battery_size'] = battery_size

		
		## Calling the method that computes the energy shared during one year

		results = shared_energy_evaluator(time_dict, input_powers_dict, technologies_dict, auxiliary_dict, fixed_analysis_flag)

		# Status of the optimisation during the current typical day
		optimisation_status = results['optimisation_status']

		# Energy produced by the PV during each month (kWh/month)
		pv_production_energy = results['pv_production_energy']

		# Energy consumed by the aggregate of households during each month (kWh/month)
		consumption_energy = results['consumption_energy']

		# Excess energy fed into the grid during each month (kWh/month)
		grid_feed_energy = results['grid_feed_energy']

		# Deficit of energy purchased from the grid during each month (kWh/month)
		grid_purchase_energy = results['grid_purchase_energy']

		# Excess energy used to charge the battery during each month (kWh/month)
		battery_charge_energy = results['battery_charge_energy']

		# Deficit of energy taken from the battery during each month (kWh/month)
		battery_discharge_energy = results['battery_discharge_energy']

		# Energy shared by the aggregate of households, according to the definition provided by the Decree-Law 162/2019 (kWh/month)
		shared_energy = results['shared_energy']


		## Processing the results

		# Monthly values

		# Self-sufficiency index in each month (%)
		self_suff_ind_month = shared_energy/(consumption_energy)*100

		# Self-consumption index in each month (%)
		self_cons_ind_month = shared_energy/(pv_production_energy)*100

		# Yearly values (kWh/year)

		pv_production_year = np.sum(pv_production_energy)

		consumption_year = np.sum(consumption_energy)

		grid_feed_year = np.sum(grid_feed_energy)

		grid_purchase_year = np.sum(grid_purchase_energy)

		battery_charge_year = np.sum(battery_charge_energy)

		battery_discharge_year = np.sum(battery_discharge_energy)

		shared_energy_year = np.sum(shared_energy) #TO CHECK

		# Self-sufficiency index in a year (%)
		self_suff_ind_year =  shared_energy_year/(consumption_year)*100

		# Self-consumption index in a year (%)
		self_cons_ind_year = shared_energy_year/(pv_production_year)*100

		#Compute KPI
		I0 = initial_investment(pv_size, battery_size) #initial investment
		cflows, energy_sales, contrib, contrib_prod = cash_flows(grid_feed_year/1000,
														     	shared_energy_year/1000,
															    pv_size, 
																beta = 0.75,
																n_years=20,
																inf_rate= 0.01
																)
		#Compute NPV and IRR
		cer_npv = NPV(cflows, 0.02, I0)
		cer_irr = IRR(cflows, I0)

		#Compute PCR
		yearly_expense_norsa = 19356
		yearly_expense_rsa = 40582
		#Community returns
		 #yearly average
		if power_max == 105.5:
			cer_pcr = PCR(yearly_expense_norsa, contrib_prod, energy_sales, I0, power_max)
		elif power_max == 140.5:
			cer_pcr = PCR(yearly_expense_norsa, contrib_prod, energy_sales, I0, power_max)
		else:
			raise RuntimeError("Power Max is not consistent")
		
		#PCR community
		#cer_pcr = PCR_community(yearly_expense_rsa, contrib - contrib_prod)
		
		#Payback time
		cer_pbt = PBT(I0, cflows)
		# In case at least one technology is subject to a parametric analysis, only yearly quantities are shown
		if fixed_analysis_flag != 1:

			# Storing the yearly values of the self-sufficiency and self-consumption indices and of the shared energy 
			# for the current configuration in order to be plotted outside of the loops

			ssi_configurations[pv_index, battery_index] = self_suff_ind_year
			sci_configurations[pv_index, battery_index] = self_cons_ind_year
			esh_configurations[pv_index, battery_index] = np.sum(shared_energy)

			# Storing the results in a tabularr form
			row = [pv_size, battery_size,
				'{0:.2f}'.format(self_suff_ind_year), 
				'{0:.2f}'.format(self_cons_ind_year), 
				'{0:.1f}'.format(np.sum(shared_energy)), 
				'{0:.1f}'.format(np.sum(pv_production_energy)), 
				'{0:.1f}'.format(np.sum(consumption_energy)), 
				'{0:.1f}'.format(np.sum(grid_feed_energy)), 
				'{0:.1f}'.format(np.sum(grid_purchase_energy)), 
				'{0:.1f}'.format(np.sum(battery_charge_energy)), 
				'{0:.1f}'.format(np.sum(battery_discharge_energy)), 
				'{0:.0f}'.format(cer_npv), #NPV
				'{0:.2f}'.format(cer_irr * 100), #IRR
				'{0:.1f}'.format(energy_sales.sum()), #Energy Sales 
				'{0:.1f}'.format(contrib_prod.sum()), #Producer Contributions
				'{0:.1f}'.format(contrib.sum() - contrib_prod.sum()), #Non-producer Contributions
				'{0:.1f}'.format(cer_pcr), #PCR
				'{}'.format(cer_pbt) #PBT
				]

			tab_results.append(row)

		# The number of configurations evaluated is updated and, in case, the progress is printed
		count_configurations += 1
		if int(count_configurations/n_configurations*100) >= perc_configurations:
			print('{:d} % completed'.format(int(count_configurations/n_configurations*100)))
			perc_configurations += 25

# Informing the user about the total time needed to evaluate the configurations
print('\n{0:d} configuration(s) evaluated in {1:.3f} s.'.format(n_configurations, toc()))



### Post-processing of the results (tabulates, .csv files and figures)


## Completing the tabulate that is going to be printed

# Header to be used in the tabulate in case of parametric analysis at least for one of the two technologies
if fixed_analysis_flag != 1:
	headers = ['PV size \n(kW)', 'Battery size \n(kWh)', 
				'SSI \n(%)', 'SCI \n(%)', 'Shared energy \n(kWh/year)', 'PV production \n(kWh/year)', 'Consumption \n(kWh/year)', \
				'Grid feed \n(kWh)', 'Grid purchase \n(kWh)', 'Battery charge \n(kWh)', 'Battery discharge \n(kWh)',
				'NPV (€)', 'IRR (%)', 'Energy Sales Next 20 Years (€)', 'Producer Contributions Next 20 Years (€)', 'Non-Producer Contributions Next 20 Years (€)',
				'PCR Yearly (%)', 'PBT (Years)'
			]

# In case of fixed size simulation for both the PV and the battery the tabulate has still to be built
# Morover, detailed results are provided (monthly)
else:
	raise RuntimeError("Please Run Parametric Analysis")

# Printing the tabulate
message = '\nOptimisation status and results (self-sufficiency and self-consumption indices, shared energy, etc.)\n'
print(message) 
print(tabulate(tab_results, headers = headers))


## Storing the tabulate in a .csv file

# Creating an /Output folder, if not already existing
dirname = 'Output'

try: Path.mkdir(basepath / dirname)
except Exception: pass

# Creating an /Output/Files folder, if not already existing
subdirname = 'Files'

try: Path.mkdir(basepath / dirname / subdirname)
except Exception: pass 

subsubdirname = 'simulation_results'

try: Path.mkdir(basepath / dirname / subdirname / subsubdirname )
except Exception: pass

fpath = basepath / dirname / subdirname / subsubdirname 

# Saving the tabulate in a .csv file

if fixed_analysis_flag != 1: filename = 'pv_{}_{}_battery_{}_{}_yearly_values.csv'.format(pv_size_range[0], pv_size_range[-1], battery_size_range[0], battery_size_range[-1])
else: filename = 'pv_{}_battery_{}_monthly_values.csv'.format(pv_size, battery_size)
 
with open(fpath / filename, mode = 'w', newline = '') as csv_file:

	header_row = [header.replace('\n', ' ') for header in headers]
	csv_writer = csv.writer(csv_file, delimiter = ';', quotechar = "'", quoting = csv.QUOTE_MINIMAL)
	csv_writer.writerow(list(header_row))
	
	for row in tab_results:
		csv_writer.writerow(list(row))

message = '\nThe detailed files about shared energy have been saved in {}.'.format(fpath)
print(message)  

#%%
#Plotter Area
import seaborn as sns
sns.set_theme() #use to reset955
sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.5)
plt.rc('figure',figsize=(16,12))
plt.rc('font', size=20)
plt.rc('lines', markersize = 18)
# =============================================================================
# plt.rc('figure',figsize=(24,16))
# plt.rc('font', size=16)
# =============================================================================
#%%
output = pd.read_csv(fpath/filename,
				   sep = ";",
				   decimal = ".",
				   encoding='cp1252')
out_irr = output['IRR (%)']
out_pcr = output['PCR Yearly (%)']

sns.scatterplot(data = output, 
				x = 'PCR Yearly (%)',
				y = 'IRR (%)',
				hue = 'Battery size  (kWh)',
				style = 'PV size  (kW)',
				palette = 'flare',
				legend = 'auto'
				)
plt.title("KPI - Optimal Design")