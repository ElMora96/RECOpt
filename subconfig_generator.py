import pandas as pd
import numpy as np
from itertools import product

class BenefitDistributionGame:
"""Class to represent benefit distribution as a game"""
#-----Nested Player class - Each player represents a participant in the REC----
	class _Player:
		"""Constructor should not be invoked by user"""
		def __init__(self, profile_we = None, profile_wd = None):
			#Store data - pd.DataFrames in usual format
			if profile_we is None or profile_wd is None:
				profile_we, profile_wd = self._simulate_profiles()
			self._profile_wd = profile_wd
			self._profile_we = profile_we
			self.player_number = None

		def _set_player_number(self, number):
			self.player_number = number

		def _generate_subconfigs(self):
			"""Generate all subconfigurations required to compute Shapley Value"""
			pass

		def _shapley_value(self):
			"""Compute shapley value of this user"""

			pass

		def _simulate_profiles(self):
			"""Run simulator to generate profiles"""
			raise NotImplementedError("Integrate Lorenti's simulation engine")


	def __init__(self):
		"""Create an instance of the game"""
		self._n_players = None #tot number
		#store value funtion results
		self._vfdb = self._create_db() #{"[00010000]":vf}

	def _create_db(self):
		"""Create database with value function result foreach configuration"""
		configs = list(product((0, 1),
								repeat = self._n_players)
								)
		configs = configs.pop(index = 0) #remove empty configuration
		#Create database
		self._vfdb = {key : self._value_function(key) for key in configs}




