import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from random import random
import matplotlib.animation as animation
from random import uniform
import time

class person:
	def __init__(self, p_surv, p_inf, p_die, status):
		self.p_surv = p_surv
		self.p_inf = p_inf
		self.p_die = p_die
		self.status = "h" #healthy by default

		self.location = None
		self.location_history = []
		self.days_been_infected = 0
		self.infection_history = []

		self.vx, self.vy = 0, 0

	def __call__(self, p_surv, p_inf, p_die, infected):
		self.p_surv = p_surv
		self.p_inf = p_inf
		self.p_die = p_die
		self.infected = infected
		return person

	def infect(person):
		if random() < person.p_inf:
			person.infected = True

	def is_infected(self):
		if self.status == "i": return True 

	def is_recovered(self):
		if self.status == "r": return True

	def is_dead(self):
		if self.status == "d": return True

	def gets_sick(self):
		self.status = "i" # infected

	def recovers(self):
		self.status = "r" # recovered

	def dies(self):
		self.status = "d" # dead

	def set_random_location(self, lim_x, lim_y):
		# gives a random location in a 2D plane
		self.location = [uniform(-lim_x, lim_x), uniform(-lim_y, lim_y)]
		self.location_history.append(self.location)
		self.infection_history.append(self.status)

	def take_random_step(self):
		x, y = randrange(-1, 2), randrange(-1, 2)
		self.location = [self.location[0] + x, self.location[1] + y]
		self.location_history.append(self.location)
		self.infection_history.append(self.status)
	
	def set_velocity(self, vx, vy):
		self.vx, self.vy = vx, vy

	def move(self, dt):
		self.location = [self.location[0] + self.vx * dt, self.location[1] + self.vy * dt]
		self.location_history.append(self.location)
		self.infection_history.append(self.status)
class pandemic:
	def __init__(self, population):
		self.country = np.empty(dtype = object, shape = (int(np.sqrt(population)), int(np.sqrt(population))))

		for i in range(len(self.country)):
			for j in range(len(self.country)):
				self.country[i,j] = person(uniform(0,.1),uniform(.1,.5), uniform(0,.3), "h" )
		# transform matrix into a list for rw simulation

		self.people = []
		self.size = 0
		s = time.time()
		for i in range(len(self.country)):
			for j in range(len(self.country)):
				self.people.append(self.country[i,j])
		print(time.time() - s)
		self.n_infected = []
		self.infected_total_day = []
		self.infected_day = []
		self.dead = []
		self.recovered = []
		self.days = 0

		self.time_evolution = []

	def infect_someone(self):
		i, j = randrange(0, len(self.country)), randrange(0, len(self.country))
		self.country[i, j].status = "i"

	def get_neighbours(self, i, j):
		nb = []
		nb.extend([(self.country[i-1,j], [i-1,j]), (self.country[i,j-1], [i,j-1]), 
			(self.country[(i+1)%len(self.country),j], [(i+1)%len(self.country),j]), 
			(self.country[i,(j+1)%len(self.country)], [i,(j+1)%len(self.country)])])
		return nb

	def local_transmission(self):
		infected = 0
		for i in range(len(self.n_infected)):
			n = self.n_infected[i][0]
			m = self.n_infected[i][1]
			nb = self.get_neighbours(n, m)
			for k in range(len(nb)):
				if random() < self.country[n,m].p_inf and (nb[k][0].status == "h" and nb[k][0].status == "d" and nb[k][0].status == "i"):
						nb[k][0].infected = True
						self.n_infected.append(nb[k][1])
						infected += 1
						# print("Got infected")
		return infected

	def recuperation(self):
		dead = 0
		recovered = 0
		indices_to_pop = []
		for i in range(len(self.n_infected)):
			n = self.n_infected[i][0]
			m = self.n_infected[i][1]
			if random() < self.country[n,m].p_surv:
				self.country[n,m].recovers()
				indices_to_pop.append(i)
				recovered += 1
			elif random() < self.country[n,m].p_die:
				self.country[n,m].dies()
				indices_to_pop.append(i)
				dead += 1
		indices_to_pop.reverse()
		for j in range(len(indices_to_pop)):
			self.n_infected.pop(indices_to_pop[j])
		return recovered, dead

	def spread_local(self, time):
		#get neighbours
		self.days = time
		t = time
		while t != 0:
			if t == time:
				for i in range(len(self.country)):
					for j in range(len(self.country)):
						if self.country[i,j].is_infected():
							self.n_infected.append([i,j])
			inf = self.local_transmission()
			recovered_day, dead_day = self.recuperation()
			self.recovered.append(recovered_day)
			self.dead.append(dead_day)
			self.infected_day.append(inf)
			self.infected_total_day.append(len(self.n_infected))
			t -= 1

	def spread_nonlocal(self, size, time, recovery_time):
		self.days = time
		self.size = size
		#each particle will do random walk
		# each particle starts at a random position in a 2D plane
		for person in self.people:
			person.set_random_location(size/2, size/2)
		#now we move each person
		for days in range(time):
			for person in self.people:

				if person.is_infected():
					person.days_been_infected += 1
				if person.days_been_infected == recovery_time:
					person.recovers() 

				person.take_random_step()
				
				# at this point we want to know if someone hits the limit of the plane.
				x = person.location[0]
				y = person.location[1]
				if abs(x) == size/2:
					person.location[0] = -x
				if abs(y) == size/2:
					person.location[1] = -y 

				# now we want to know who bumps into each other.
			for person1 in self.people:
				for person2 in self.people:
					if( person1.location == person2.location
															 and person1 !=  person2 
															 and (person1.is_infected() or person2.is_infected())
															 and not(person1.is_recovered() or person2.is_recovered()) ):
						# a transmission may occur.
						if person1.is_infected():
							# here we could use a different p of infection but for now we keep it constant
							person2.gets_sick()
						elif person2.is_infected():
							person1.gets_sick()

	def spread_nonlocal_deterministic(self, size, t, recovery_time):
		start = time.time()
		self.days = t
		self.size = size
		#each particle will do random walk
		# each particle starts at a random position in a 2D plane
		for person in self.people:
			person.set_random_location(size/2, size/2)
			person.set_velocity(uniform(-3,3), uniform(-3,3))
		#now we move each person
		for days in range(t):

			inf, rec, dead, is_infected = 0, 0, 0, 0
			for person in self.people:

				if person.is_infected():
					person.days_been_infected += 1
					is_infected += 1
				if person.days_been_infected == recovery_time:
					person.recovers()
					rec += 1 
				if random() < 0.01 and person.is_infected():
					person.dies()
					dead += 1

				person.move(0.05)
				
				# at this point we want to know if someone hits the limit of the plane.
				if abs(person.location[0]) >= size/2:
					person.vx = -person.vx
				if abs(person.location[1]) >= size/2:
					person.vy = -person.vy 

				# now we want to know who bumps into each other.
			for person1 in self.people:
				for person2 in self.people:
					# distance = np.linalg.norm([person1.location[0] - person2.location[0], person1.location[1] - person2.location[1]] )
					distance2 = ((person1.location[0] - person2.location[0])**2 +  (person1.location[1] - person2.location[1])**2)

					if( distance2 < 0.02 and person1 !=  person2 
										 and (person1.is_infected() or person2.is_infected())
										 and not(person1.is_recovered() or person2.is_recovered()) ):
						# a transmission may occur.

						if person1.is_infected():
							# here we could use a different p of infection but for now we keep it constant
							person2.gets_sick()
							inf += 1
						elif person2.is_infected():
							person1.gets_sick()
							inf += 1
			self.infected_day.append(inf)
			self.infected_total_day.append(is_infected)
			self.recovered.append(rec)
			self.dead.append(dead)	
		print( "In spreading: " + str(time.time() -start))
	
	def animation_nonlocal(self):
		locations_history = []
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_ylim(-self.size/2 * 1.1, self.size/2 * 1.1)
		ax.set_xlim(-self.size/2 * 1.1, self.size /2* 1.1)
		ims = []
		start = time.time()
		for days in range(self.days):
			x, y, color = [], [], []
			for person in self.people:
				x.append(person.location_history[days][0])
				y.append(person.location_history[days][1])

				if person.infection_history[days] == "h":
					color.append("Blue")
				elif person.infection_history[days] == "i":
					color.append("Red")
				elif person.infection_history[days] == "r":
					color.append("Green")
				else:
					color.append("Black")
				# for person in self.people:
		
			# 	if person.infection_history[days]:
			# 		color.append("Red")
			# 	else:
			# 		color.append("Blue")
			ims.append([ax.scatter(x, y, color = color)])
		
		ani = animation.ArtistAnimation(fig, ims, interval= 30, blit=True,
	                                	repeat_delay=1000) 
		print("In creatin animation: " + str(time.time() - start))
		plt.show()
			

		

	def statistics_local(self):
			fig  = plt.figure()
			ax = fig.add_subplot(1,1,1)
			ax.plot(np.arange(0, self.days, 1), self.infected_day, color = 'red', label = ' ' )
			ax.set_title("Number of infections registered per day")
			ax.legend()

			fig2 = plt.figure()
			ax2 = fig2.add_subplot(1,1,1)
			ax2.plot(np.arange(0, self.days, 1), self.infected_total_day, label = "a label")
			ax2.set_title("Number of people infected each day")
			ax2.legend()

			fig3 = plt.figure()
			ax3 = fig3.add_subplot(1,1,1)
			ax3.bar(np.arange(0, self.days, 1), self.recovered)
			ax3.set_title("Number of people recovered per day")
			
			plt.show()

			fig4 = plt.figure()
			ax = fig4.add_subplot(1,1,1)
			xdata = np.arange(0, self.days, 1)
			ydata = self.infected_total_day
			ax.bar(xdata, ydata, color = 'red', label = ' ' )
			ax.set_title("Number of infections registered per day")
			ax.legend()
			ax3 = fig4.add_subplot(1,1,1)
			ax3.bar(np.arange(0, self.days, 1),self.recovered, color = 'green')
			ax.set_title("Number of people recovered per day")
			ax4 = fig4.add_subplot(1,1,1)
			ax.bar(np.arange(0, self.days, 1), self.dead)

			plt.show()
# country

# generate infection and spread it
def init_disease(population):
	country = np.ones(shape = (int(np.sqrt(population)),int(np.sqrt(population))))
	# 1 means healthy, -1 means infected, 0 means dead, 2 means recovered
	# those cells with value = 2 can't be infected again. Same those who have value = 0 and, thus, are dead. XD
	seed_i = randrange(0,len(country))
	seed_j = randrange(0, len(country))
	# that person becomes infected.
	country[seed_i, seed_i] = -1
	return country

def spread(country, p_trans, p_die,days, p_surv):# days is just the number of iterations
	#now transmission can take place
	country_evolution_ims = []
	d = days
	healthy = 0
	dead = 0
	healthy_len = []
	infected_len= []
	dead_len = []
	days_list = []
	infected = []
	while d != 0:
		if d == days:
			for k in range(len(country)):
				for s in range(len(country)):
					if country[k,s] == -1:
						infected.append([k,s])

		for n in range(len(infected)):
			try:
				he, de = transmission(infected, n, country, infected[n][0], infected[n][1] ,p_trans, p_die, p_surv)
				healthy += he
				dead += de
			except(IndexError):
				pass
		
		days_list.append(d)
		if len(infected) == 0:
			pass
			# print("Virus erradicated")
		d -= 1
		infected_len.append(len(infected))
		healthy_len.append(healthy)
		dead_len.append(dead)
		country_evolution_ims.append([plt.imshow(country,animated = True)])
	
	return country, infected_len, days_list, healthy_len, dead_len, country_evolution_ims


def transmission(infected, n, country, i, j, p_trans, p_die, p_surv): 
	x = country[i-1,j]
	y = country[i,j-1]
	z = country[(i+1)%len(country),j]
	t = country[i,(j+1)%len(country)] 

	healthy = 0
	dead = 0

	if random() < p_trans and x == 1:
		country[i-1,j] = -1
		infected.append([i-1,j])
	if random() < p_trans and y == 1:
		country[i,j-1] = -1
		infected.append([i,j-1])
	if random() < p_trans and z == 1:
		country[(i+1)%len(country),j] = -1
		infected.append([(i+1)%len(country),j])
	if random() < p_trans and t == 1: 
		country[i,(j+1)%len(country)] = -1
		infected.append([i,(j+1)%len(country)]) 
	# host can die
	if random() < p_die:
		country[i, j] = 0
		infected.pop(n)
		dead += 1
	elif random() < p_surv:
		country[i, j] = 2
		infected.pop(n)
		healthy += 1

	return healthy, dead

def coronavirus_molts():
	p_inf = 0.05
	c_change = 0
	for i in range(20):
		country = init_disease(int(1e5))
		days = 1250
		p_inf += 0.025
		p_die = 0.03
		p_surv = 0.01
		country, inf, d, healthy, dead, imgs = spread(country, p_inf, p_die, days, p_surv)
		d.reverse()
		# plt.matshow(country)
		# plt.colorbar()
		# plt.show()
		c_change += 0.04
		plt.plot(d, inf, label ="Infectivitat: " + str(round(p_inf * 100,1)) +"%, Mortalitat:" + str(round(p_die*100, 1)) + "%",
						color = (1 - c_change,0.,0.) )
		# plt.plot(d, healthy, color = "green")
		# plt.plot(d, dead)
		plt.xlabel("Dies")
		plt.ylabel("Població infectada")
		plt.legend()
		print(i + 1)
		# plt.text(300,1500,"Infectivitat: " + str(p_inf * 100) +"%, Mortalitat:" + str(p_die*100) + "%,")


	plt.show()

def evolucio_temporal_covid(population,days):
	fig = plt.figure()

	country = init_disease(population)
	country, inf, d, healthy, dead, country_evolution_list = spread(country, 0.75, 0.04, days, 0.1)


	ani = animation.ArtistAnimation(fig, country_evolution_list, interval=50, blit=True,
	                                repeat_delay=1000) 

	plt.show()
	plt.imshow(country)
	plt.colorbar()
	plt.show()

coronavirus_molts()
# evolucio_temporal_covid(int(1e5), 100)



# 			 LOCAL 

# covid19 = pandemic(population = int(1e6))
# for i in range(int(1e5)):
# 	covid19.infect_someone()
# covid19.spread_local(time=100)
# covid19.statistics_local()

#			NON LOCAL
# covid19 = pandemic(population = 100)
# for i in range(10):
# 	covid19.infect_someone()
# covid19.spread_nonlocal_deterministic(size = 10, t = int(600), recovery_time = 200)
# covid19.animation_nonlocal()
# covid19.statistics_local()





