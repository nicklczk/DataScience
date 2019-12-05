import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import csv
mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
axyz = fig.gca(projection='3d')
plt.axis('equal')


# Income dataset
county_income = [] #from csv

county_income_median = []

with open("county_median_income3.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        county_income.append((str(row[0])))
        county_income_median.append([str(row[0]),float(row[1])])

county_avg_income_median = []

unique_counties_income = (list(set(county_income))) 

for county in unique_counties_income:
	total = 0
	frequency = 0
	for row in county_income_median:
		if (row[0] == county):
			total = total + row[1]
			frequency = frequency + 1		
	average = total / frequency
	county_avg_income_median.append([county, average])


for row in county_avg_income_median:
	print (row)

with open("county_avg_income_median3.csv","w+") as csv_file:  
    for x in county_avg_income_median:
        row = str(x[0]) +","+ str(x[1])
        csv_file.write(row+'\n')	

#Heart disease dataset
county_heart = [] #from csv

county_heart_disease = []

with open("county_heart_disease3.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        county_heart.append((str(row[0])))
        county_heart_disease.append([str(row[0]),float(row[1])])

county_avg_heart_disease = []

unique_counties_heart = (list(set(county_heart))) 

for county in unique_counties_heart:
	total = 0
	frequency = 0
	for row in county_heart_disease:
		if (row[0] == county):
			total = total + row[1]
			frequency = frequency + 1
	average = total / frequency
	county_avg_heart_disease.append([county, average])


for row in county_avg_heart_disease:
	print (row)

with open("county_avg_heart_disease3.csv","w+") as csv_file:  
    for x in county_avg_heart_disease:
        row = str(x[0]) +","+ str(x[1])
        csv_file.write(row+'\n')	

# saving the data for common counties

with open("common_county_data2.csv","w+") as csv_file:  
	for county in unique_counties_heart:
		if (county in unique_counties_income):
			income = 0
			heart = 0 
			for x in county_avg_income_median:
				if (x[0] == county):
					income = x[1]
			for x in county_avg_heart_disease:
				if(x[0] == county):
					heart = x[1]
			row = str(county)+ "," + str(income) +","+ str(heart)
			csv_file.write(row+'\n')							