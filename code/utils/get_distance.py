def get_distance(v1, v2):
	total = 0
	for a,b in zip(v1,v2):
		total += (a-b)**2
	return total**0.5