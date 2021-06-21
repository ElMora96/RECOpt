import pandas as pd
import numpy as np
from itertools import product
import glob 
from copy import deepcopy
from math import factorial

class BenefitDistributionGame:
"""Class to represent benefit distribution as a game"""

#-----Nested Player class - Each player represents a participant in the REC----
	class _Player:
		"""Constructor should not be invoked by user"""
		def __init__(self, player_number, profile_we = None, profile_wd = None):
			"""Parameters:
				player_number: int. 
			"""
			#Store data - pd.DataFrames in usual format.
			if profile_we is None or profile_wd is None:
				profile_we, profile_wd = self._simulate_profiles()
			self._profile_wd = profile_wd
			self._profile_we = profile_we

			#Assign player number
			self.player_number = player_number

			#Insert PV Power and Battery Size for this user.
			self._pv_power = input("Insert PV Power for this player: ")
			if self._pv_power == '': 
				self._pv_power = 0
			else:
				self._pv_power = int(self._pv_power)
			self._battery_size = input("Insert battery size for this player: ")
			if self._battery_size == '':
				self._battery_size = 0
			else:
				self._battery_size = int(self._battery_size)

		def shapley_value(self, vfdb):
			"""Compute shapley value of this user.
			Parameters:
			vfdb: value function database (dict)
			Returns:
			shapley: float. Shapley value for this player
			"""
			n = len(vfdb)
			#Compute all configurations excluding this players
			configs_without = [key for key in vfdb if key[self.player_number == 0]]
			#Parallel configurations including this player
			configs_with = deepcopy(configs_without)
			for conf in configs_with:
				conf[self.player_number] = 1
			#Compute Shapley value
			shapley = 0
			for cwith, cwout in zip(configs_with, configs_without):
				s = sum(cwithout) #Number of elements in subconfiguration
				weight = factorial(s)*factorial(n - s - 1)*(1/factorial(n))
				term = vfdb[cwith] - vfdb[cwout] #Access stored values
				shapley += weight*term
			return shapley

		def _simulate_profiles(self):
			"""Run simulator to generate profiles"""
			raise NotImplementedError("Integrate Lorenti's simulation engine")

#Game Class - Private Methods
	def __init__(self, players_path):
		"""Create an instance of the game.
		Parameters.
		players_path: path to folder where players data are stored"""
		self._n_players = None #tot number
		#Create player list
		self.players = self._create_players(players_path)
		#store value funtion results

	def _create_db(self):
		"""Create database with value function result foreach configuration"""
		#Create all configuration
		configs = list(product((0, 1),
								repeat = self._n_players)
								)
		#Create database
		#Here use dictionary comprehension; eventually use a starmap
		db = {key : self._value_function(key) for key in configs}
		return db

	def _value_function(self, config):
		"""Compute value of given config. Wrapper for
		Lorenti's module.
		Parameters:
		config: binary iterable of length self._n_players
		Returns:
		float, positive value of config.
		"""
		#If configuration is empty, value is zero
		if all([pos == 0 for pos in config]):
			return 0 #No consumption -> No shared energy
		profile_wd, profile_we, pv_size, battery_size = self._subconfig_inputs(config)
		if pv_size == 0:
			return 0 #No shared energy, nor power sold
		raise NotImplementedError("Integrate Lorenti's Module")	
		
	def _create_players(self, players_path):
		"""Create players for game.
		Parameters:
		Players_path. Path were player data are stored
		(in .csv format).
		Returns:
		List of _Player objects
		"""
		#weekday profiles
		wdlist = glob.glob( (players_path / "wd" / "*.csv"))
		#weekend profiles
		welist = glob.glob( (players_path / "wd" / "*.csv"))
		#Create players
		player_list = []
		for ix, wd, we in enumerate(zip(wdlist, welist)):
			wd_data = pd.read_csv(wd,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]
			we_data = pd.read_csv(we,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]
			#Instantiate new player
			newplayer = _Player(ix, wd_data, we_data)
			player_list.append(newplayer)

		return player_list

	def _subconfig_inputs(config):
		"""Generate all subconfiguration inputs.
		Parameters:
		config: binary iterable.
		Returns:
		profile_wd: np.array of shape (24,12)
		profile_we: np.array of shape (24,12)
		pv_size: float
		battery_size : float
		"""
		profile_wd = np.zeros((24,12))
		profile_we = np.zeros((24,12))
		pv_size = 0
		battery_size = 0
		sublist = [player for ix, player in enumerate(self.players) if config[ix] == 1]
		for player in sublist:
			profile_wd += player._profile_wd
			profile_we += player._profile_we
			pv_size += player._pv_size
			battery_size += player._battery_size

		return profile_wd, profile_we, pv_size, battery_size

		
#Game Class - Public Methods
	
	def play(self):
		"""Run Game.
		Return:
		shapley_vals: list with shapley value for each player"""
		#Create value function database
		self._vfdb = self._create_db() #{"[00010000]":vf}
		#Compute Shapley values
		shapley_vals = [player.shapley_value(self._vfdb) for player in self.players]
		return shapley_vals


