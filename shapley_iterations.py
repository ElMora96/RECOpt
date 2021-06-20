from math import  comb
def factorial(n):
	if n == 0 or n == 1:
		return 1
	else:
		return n*factorial(n-1)

def shapley_niter(N):
	"""N: number of participants to REC.
	Returns number of iterations required 
	to compute shapley value."""
	total = 0
	for k in range(1, N):
		add = comb(N - 1, k)
		total += add
	return total

def approx_shapley_niter(N, delta, epsilon):
	num = N*(N-1)
	den = delta * (epsilon**2)
	return num/den


##Unit test##

if __name__ == '__main__':
	num = 30
	print("Normal: ", shapley_niter(num))
	print("Physics: ", 2**(num - 1) - 1)
	print("Approximation: ", approx_shapley_niter(num, 0.1,0.1))