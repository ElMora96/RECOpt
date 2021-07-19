import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import sys 
# The basepath of the file is stored in a variable 
basepath = Path(__file__).parent
sys.path.append(basepath.as_posix())
import pandas as pd
from scipy.interpolate import interp1d
from tabulate import tabulate
import parameters_input as inp
import plot_generator as plot
import datareader
from tictoc import tic, toc
from shared_energy_evaluator import shared_energy_evaluator
from kpi_rev import *
import seaborn as sns

sns.set_theme() #use to reset955
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale= 2.5)
plt.rc('figure',figsize=(16,12))
plt.rc('font', size=20)
plt.rc('lines', markersize = 10) #18 for cv plots

# Palette of colors to be used for plotting the results
colors = [(230, 25, 75),
		(60, 180, 75),
		(255, 225, 25),
		
		(0, 130, 200),
		(245, 130, 48),
		(145, 30, 180),
		(70, 240, 240),
		(240, 50, 230),
		(210, 245, 60),
		(250, 190, 212),
		(0, 128, 128),
		(220, 190, 255),
		(170, 110, 40),
		(255, 250, 200),
		(128, 0, 0),
		(170, 255, 195),
		(128, 128, 0),
		
		(255, 215, 180),
		(0, 0, 128),
		(128, 128, 128)]

# Transforming into rgb triplets
colors_rgb = []
for color in colors:
	color_rgb = []
	for value in color:
		color_rgb.append(float(value)/255) 
	colors_rgb.append(tuple(color_rgb))


###############################################################################

### Parameters setup

# Contractual power (MAX) (kW)
power_max = 20#105.5 - 140.5 20 solo palestra

# Time-step used to aggregated the results (min): 1 | 5 | 10 | 15 | 10 | 30 | 45 | 60 
# FIXED PARAM : 60 minutes
dt_aggr = 60 


## Type of simulation
# The type of simulation is chosen (fixed size or paramtric analysis) both for the pv
# and the battery and the ranges for the sizes are specified as well. This is done using the method
# simulation_setup from parameters_input.py (inp) that returns a dictionary and a list.

# Photovoltaic (PV)
pv_setup, pv_size_range = inp.simulation_setup('PV')

# Battery 
battery_setup, battery_size_range = inp.simulation_setup('battery')


message = '\nNo consumption simulation is performed. Data are provided by User.'
print(message) 


# Simulation setup (PV)
tab = []
for param in pv_setup:
	row = [param, pv_setup[param]]
	tab.append(row)

message = '\nSimulation setup (PV)'
print(message) 
print(tabulate(tab, headers=['Parameter', 'Value']))

# Simulation setup (battery)

tab = []
for param in battery_setup:
	row = [param, battery_setup[param]]
	tab.append(row)

message = '\nSimulation setup (battery)'
print(message) 
print(tabulate(tab, headers=['Parameter', 'Value']))


## Building seasons, months and week dictionaries 
# This is done in order to explore different seasons/months and day-types (week-day/weekend-day)
# during the simulations (id numbers and nicknames are needed to find the various files that are 
# to be uploaded and then find the correct positions in arrays for each data)
 
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

# While the routine that evaluates the load profiles works with typical profiles for each season, 
# the optimization procedure works with tyipcal profiles for each month.  
# A reference year is considered, in which the first day (01/01) is a monday. 
# Therefore, conventionally considering that winter lasts from january to march 
# spring from april to june, summer from july to september and autumn from october
# to december, each month has got the following number of weekdays and weekend days.

# Creating a dictionary for the months
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

# Storing all these dictionaries in a new dict that is passed to the various methods
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


## Time discretization
# Time is discretized in steps of one hour (according to the definition of shared energy in the Decree Law 162/2019 - "Mille proroghe")

# Total time of simulation (h) - for each typical day
time = 24

# Timestep for the simulation (h) (the keyboard input dt_aggr is given in minutes)
#Fixed timestep - 1h
dt = 1 #dt_aggr/60

# Vector of time, from 00:00 to 23:59, i.e. 24 h
time_sim = np.arange(0, time, dt)
time_length = np.size(time_sim)

# Storing all the elements needed for the time-discretization in a dictionary that is passed to the various methods
time_dict = {
	'time': time,
	'dt': dt,
	'time_sim': time_sim,
	'time_length': time_length,
	}



### Input data

# Maximum power from the grid (total) (kW)
grid_power_max = power_max


## Battery specification
# The battery specifications are stored in a file that can be read using the method read_param
# from the module datareader.py, that will return a dictionary

battery_specs = datareader.read_param('battery_specs.csv', ';', 'Input')

# Storing the information about the various technologies considered in a dictionary
# that is passed to the various methods
technologies_dict = {
	'grid_power_max': grid_power_max,
	'battery_specs': battery_specs,
	}


## Unit production from the photovoltaic installation (kWh/h/kWp)
# The unit production during each hour (kWh/h/kWp) from the photovoltaic installation can 
# be read using the method read_general from the module datareader.py, that returns a 2d-array
# containing the time vector on the first columns and the unit production during each hour in each
# month in the other columns

#Little ugly
import pvgis_to_csv #runs PVGIS Data transformation
data_pv = datareader.read_general('pv_production_unit.csv', ';', 'Input')

time_pv = data_pv[:, 0]
pv_production_unit = data_pv[:, 1:]

# Interpolating the unit pv production, if it has a different time resolution
if (time_pv[-1] - time_pv[0])/(np.size(time_pv) - 1) != dt:
	
	f_pv = interp1d(time_pv, pv_production_unit, kind = 'linear', axis = 0, fill_value = 'extrapolate')
	pv_production_unit = f_pv(time_sim)


## Consumption from the aggregate of PODs

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

### Energy shared from the aggregate of PODs during one year, for all possible configurations

# Total number of configurations to be analysed
n_pv_sizes = len(pv_size_range)
n_battery_sizes = len(battery_size_range)
n_configurations = n_pv_sizes*n_battery_sizes

# Creating a flag that is used to assess if a fixed analysis is performed on both technologies
# In this case detailed results are stored in .csv files and figures
fixed_analysis_flag = 0
if n_configurations == 1: fixed_analysis_flag = 1

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
		# For each typical day (two for each month, i.e. 24 typical days) the operation of the battery
		# is optimised using a MILP procedure in order to minimize the exchanges with the grid
		# The method that evaluates the shared energy returns a series of arrays having the same size
		# as the number of months, where values for the energy (shared, consumed, produced, etc.) are stored

		results = shared_energy_evaluator(time_dict, input_powers_dict, technologies_dict, auxiliary_dict, fixed_analysis_flag)

		# Storing the results

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


		## Post-processing of the results 
		# If different configuration are considered, some quantities (iss, isc, esh) are stored for each configuration
		# in order to be plotted outside of the loops (that iterate among configurations). The tabulate is instead built 
		# inside the loops to avoid creating too many variable for storing the results.
		# If only one configuration is considered the tabulate can be created outside the loop without any change on the final result

		# In case at least one technology is subject to a parametric analysis, only yearly quantities are shown
		if fixed_analysis_flag != 1:

			# Storing the yearly values of the self-sufficiency and self-consumption indices and of the shared energy 
			# for the current configuration in order to be plotted outside of the loops

			ssi_configurations[pv_index, battery_index] = self_suff_ind_year
			sci_configurations[pv_index, battery_index] = self_cons_ind_year
			esh_configurations[pv_index, battery_index] = np.sum(shared_energy)

			# Storing the results in a tabularr form

			# List of days in which the optimisation did not work or was infeasible
			opt_did_not_work_list = []
			# opt_unnecessary_list = []
			opt_infeasible_list = []

			# Optimisation status
			for month in months:
				month_nickname = months[month]['id'][1]

				for day in days:
					day_nickname = days[day][1]

					opt_status = optimisation_status[month][day].lower().strip('\'",._- ').replace('.','').replace(' ','_')
					if opt_status == 'opt_did_not_work': opt_did_not_work_list.append('{}, {}'.format(month_nickname.capitalize(), day_nickname))
					# elif opt_status == 'opt_unnecessary': opt_unnecessary_list.append('{}, {}'.format(month_nickname.capitalize(), day_nickname))
					elif opt_status == 'infeasible': opt_infeasible_list.append('{}, {}'.format(month_nickname.capitalize(), day_nickname))

			# Energy values in a year (kWh/year)
			row = [pv_size, battery_size, '\n'.join(opt_did_not_work_list), '\n'.join(opt_infeasible_list), \
				'{0:.2f}'.format(self_suff_ind_year), \
				'{0:.2f}'.format(self_cons_ind_year), \
				'{0:.1f}'.format(np.sum(shared_energy)), \
				'{0:.1f}'.format(np.sum(pv_production_energy)), \
				'{0:.1f}'.format(np.sum(consumption_energy)), \
				'{0:.1f}'.format(np.sum(grid_feed_energy)), \
				'{0:.1f}'.format(np.sum(grid_purchase_energy)), \
				'{0:.1f}'.format(np.sum(battery_charge_energy)), \
				'{0:.1f}'.format(np.sum(battery_discharge_energy)), \
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
	headers = ['PV size \n(kW)', 'Battery size \n(kWh)', 'Opt. did not work in\n*values fixed', 'Opt. infeasible in\n*minor issue', \
				'SSI \n(%)', 'SCI \n(%)', 'Shared energy \n(kWh/year)', 'PV production \n(kWh/year)', 'Consumption \n(kWh/year)', \
				'Grid feed \n(kWh)', 'Grid purchase \n(kWh)', 'Battery charge \n(kWh)', 'Battery discharge \n(kWh)']

# In case of fixed size simulation for both the PV and the battery the tabulate has still to be built
# Morover, detailed results are provided (monthly)
else:

	# Storing the results in a tabular form 

	headers = ['Month', 'Week-day', 'Weekend-day', 'SSI \n(%)', 'SCI \n(%)', \
		'Shared energy \n(kWh)', 'PV production \n(kWh)', 'Consumption \n(kWh)', \
		'Grid feed \n(kWh)', 'Grid purchase \n(kWh)', 'Battery charge \n(kWh)', 'Battery discharge \n(kWh)']
	
	# Monthy results
	for month in months:
		mm = months[month]['id'][0]
		row = [month.capitalize()]

		# Monthy results: optimisation status for each typical day
		for day in days:
			row = row + [optimisation_status[month][day]]

		# Montlhy results: energy values (kWh/month) and performance indices (%)
		row = row + (['{0:.2f}'.format(self_suff_ind_month[mm]), \
					'{0:.2f}'.format(self_cons_ind_month[mm]), \
					'{0:.1f}'.format(shared_energy[mm]), \
					'{0:.1f}'.format(pv_production_energy[mm]), \
					'{0:.1f}'.format(consumption_energy[mm]), \
					'{0:.1f}'.format(grid_feed_energy[mm]), \
					'{0:.1f}'.format(grid_purchase_energy[mm]), \
					'{0:.1f}'.format(battery_charge_energy[mm]), \
					'{0:.1f}'.format(battery_discharge_energy[mm]), \
					])

		tab_results.append(row)

	# Yearly results: energy values (kWh/year) and performance indices (%)
	tab_results.append([]*len(headers))
	tab_results.append(['Year', '/', '/', \
				'{0:.2f}'.format(self_suff_ind_year), \
				'{0:.2f}'.format(self_cons_ind_year), \
				'{0:.1f}'.format(shared_energy_year), \
				'{0:.1f}'.format(pv_production_year), \
				'{0:.1f}'.format(consumption_year), \
				'{0:.1f}'.format(grid_feed_year), \
				'{0:.1f}'.format(grid_purchase_year), \
				'{0:.1f}'.format(battery_charge_year), \
				'{0:.1f}'.format(battery_discharge_year), \
				])


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


## Creating and storing figures

dirname = 'Output'

# Creating an /Output/Figures folder, if not already existing
subdirname = 'Figures'

try: Path.mkdir(basepath / dirname / subdirname)
except Exception: pass

##################################
########   PLOTS  ################
##################################

# Different figures are created in case of fixed analysis for both the pv and the battery
# or in case of at least one parametric analysis

# If there is at least one parametric analysis, figures showing the trend of the iss, isc and shared energy
# depending on the sizes are created
if fixed_analysis_flag != 1:

	subsubdirname = 'shared_energy_results_pv_{}_{}_battery_{}_{}'.format(pv_size_range[0], pv_size_range[-1], battery_size_range[0], battery_size_range[-1])

	try: Path.mkdir(basepath / dirname / subdirname / subsubdirname )
	except Exception: pass

	fpath = basepath / dirname / subdirname / subsubdirname 

	# The method parametric_analysis from plot_generator.py takes a main_size and a lead_size. A number of subplots will be
	# created depending on the sizes present in the lead_size_range, while the sizes in the main_size_range will be used as x axis

	# The data to be plotted are the iss, isc and shared energy for each configuration
	plot_specs = {
		0: {'type': 'plot', 'yaxis': 'right', 'label': 'SSI'},
		1: {'type': 'plot', 'yaxis': 'right', 'label': 'SCI'},
		2: {'type': 'bar', 'yaxis': 'left', 'label': 'Shared energy'},
		}

	fig_specs = {
		'suptitle': None,
		'xaxis_label': 'PV size (kW)',
		'yaxis_right_label': 'Performance index (%)',
		'yaxis_left_label': 'Energy (kWh/year)',
		'lead_size_name': 'Battery',
		'lead_size_uom': 'kWh',
		'yaxis_right_ylim': [0, 1.1*np.max(np.maximum(ssi_configurations, sci_configurations))],
		'yaxis_left_ylim': [0, 1.1*np.max(esh_configurations)],
		}

	# If the analysis is parametric on the pv size, a plot is generated for each battery size,
	# showing the trend of iss, isc and shared energy with the pv size
	if n_pv_sizes > 1:
		for battery_size in battery_size_range:

			battery_index = battery_size_range.index(battery_size)
			data  = np.stack((ssi_configurations[:, battery_index], \
							sci_configurations[:, battery_index], \
							esh_configurations[:, battery_index]), axis = 1)
			data = data[:, np.newaxis, :]

			fig = plot.parametric_analysis(pv_size_range, [battery_size], data, plot_specs, fig_specs)
		
			filename = 'battery_{}_pv_{}_{}_parametric_analysis.png'.format(battery_size, pv_size_range[0], pv_size_range[-1])  
			fig.savefig(fpath / filename) 

	# If the analysis is parametric on the battery size, a plot is generated for each pv size,
	# showing the trend of iss, isc and shared energy with the battery size
	if n_battery_sizes > 1:
		for pv_size in pv_size_range:

			pv_index = pv_size_range.index(pv_size)
			data  = np.stack((ssi_configurations[pv_index, :], \
							sci_configurations[pv_index, :], \
							esh_configurations[pv_index, :]), axis = 1)
			data = data[np.newaxis, :, :]

			fig_specs['xaxis_label'] = 'Battery size (kWh)'
			fig_specs['lead_size_name'] = 'PV'
			fig_specs['lead_size_uom'] = 'kW'
			data = np.transpose(data, axes = (1, 0, 2))

			fig = plot.parametric_analysis(battery_size_range, [pv_size], data, plot_specs, fig_specs)
		
			filename = 'pv_{}_battery_{}_{}_parametric_analysis.png'.format(pv_size, battery_size_range[0], battery_size_range[-1])  
			fig.savefig(fpath / filename) 
			# plt.close(fig)
 
	# A parametric chart is generated where for the pairs of PV/battery sizes are represented basing on the values of isc and iss
	fig_specs = {
		'suptitle': 'Parametric analysis on the PV and/or battery size',
		'xaxis_label': 'Self-Consumption Index (%)',
		'yaxis_label': 'Self-Sufficiency Index (%)',
		'xaxis_lim': [0.98*np.min(sci_configurations), 1.02*np.max(sci_configurations)],
		'yaxis_lim': [0.95*np.min(ssi_configurations), 1.05*np.max(ssi_configurations)],
		}

	plot_specs = {}

	for battery_size in battery_size_range:

		battery_index = battery_size_range.index(battery_size)
		plot_specs[battery_index] = {'plot_xvalues': sci_configurations[:, battery_index], 'plot_yvalues': ssi_configurations[:, battery_index], \
			'plot_yaxis': 'left', 'plot_label': 'Battery: {} kWh'.format(battery_size), 'plot_linestyle': '-', 'plot_marker': ''}
	
	for pv_size in pv_size_range:

		pv_index = pv_size_range.index(pv_size)
		plot_specs[pv_index + battery_index + 1] = {'plot_xvalues': sci_configurations[pv_index, :], 'plot_yvalues': ssi_configurations[pv_index, :], \
			'plot_yaxis': 'right', 'plot_label': 'PV: {} kW'.format(pv_size), 'plot_linestyle': '--', 'plot_marker': 's'}

	fig = plot.parametric_chart(plot_specs, fig_specs)

	filename = 'pv_{}_{}_battery_{}_{}_parametric_chart.png'.format(pv_size_range[0], pv_size_range[-1], battery_size_range[0], battery_size_range[-1])  
	fig.savefig(fpath / filename) 
	# plt.close(fig)


		


# If the analysis is fixed-size on both the PV and the battery, detailed results (power fluxes)
# are showed in the figures
else:

	subsubdirname = 'shared_energy_results_pv_{}_battery{}'.format(pv_size, battery_size)

	try: Path.mkdir(basepath / dirname / subdirname / subsubdirname )
	except Exception: pass

	fpath = basepath / dirname / subdirname / subsubdirname 

	pv_production_month_day = results['pv_production_month_day']
	consumption_month_day = results['consumption_month_day']
	grid_feed_month_day = results['grid_feed_month_day']
	grid_purchase_month_day = results['grid_purchase_month_day']
	battery_charge_month_day = results['battery_charge_month_day']
	battery_discharge_month_day = results['battery_discharge_month_day']
	battery_energy_month_day = results['battery_energy_month_day']
	shared_power_month_day = results['shared_power_month_day']

	# Plotting the quantities for each month

	plot_specs = {
		0: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'pv_production', 'plot_color': colors_rgb[0], 'plot_linestyle': '-', 'plot_marker': 's'},
		1: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'consumption', 'plot_color': colors_rgb[3], 'plot_linestyle': '-', 'plot_marker': 'o'},
		2: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'grid_feed', 'plot_color': colors_rgb[2], 'plot_linestyle': '-', 'plot_marker': ''},
		3: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'grid_purchase', 'plot_color': colors_rgb[4], 'plot_linestyle': '-', 'plot_marker': 's'},
		4: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'battery_charge', 'plot_color': colors_rgb[6], 'plot_linestyle': '--'},
		5: {'plot_type': 'plot', 'plot_yaxis': 'left', 'plot_label': 'battery_discharge', 'plot_color': colors_rgb[7], 'plot_linestyle': '--'},
		6: {'plot_type': 'bar', 'plot_yaxis': 'left', 'plot_label': 'shared_power', 'plot_color': colors_rgb[9], 'plot_alpha': 0.5},
		7: {'plot_type': 'plot', 'plot_yaxis': 'right', 'plot_label': 'battery SOC', 'plot_color': colors_rgb[11], 'plot_linestyle': '--'},
	}

	fig_specs = {
		'suptitle': '\nCER - Power Fluxes',
		'xaxis_label': 'Time (h)',
		'yaxis_left_label': 'Power (kW)',
		'yaxis_right_label': 'SOC (%)',
	}
	

	for month in months:

		mm = months[month]['id'][0]

		fig_specs['title'] = month

		# Battery's SOC is plotted; to avoid division by 0 in case of battery_size == 0:
		batt_size = max(1e-5, battery_size)
		
		powers = np.stack((pv_production_month_day[:, mm, :],
						consumption_month_day[:, mm, :],
						grid_feed_month_day[:, mm, :],
						grid_purchase_month_day[:, mm, :],
						battery_charge_month_day[:, mm, :],
						battery_discharge_month_day[:, mm, :],
						shared_power_month_day[:, mm, :],
						battery_energy_month_day[:, mm, :]/batt_size*100),
						axis = 0)

		fig = plot.daily_profiles(time_sim, powers, plot_specs, fig_specs)


		filename = 'power_fluxes_{}_{}.png'.format(mm, month)
		fig.savefig(fpath / filename)
		# plt.close(fig)


message = '\nThe detailed figures about shared energy have been saved in {}.'.format(fpath)
print(message)     

print('\nEnd. Total time: {0:.2f} s.'.format(toc()))
plt.show()










