import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import csv
mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
axyz = fig.gca(projection='3d')
plt.axis('equal')


# Median Income dataset
county_income = [] #from csv

income_range = [] # from csv

income_median = [] #calculated by taking endpoint for ranges


with open("NYSERDA_Low-_to_Moderate-Income_New_York_State_Census_Population_Analysis_Dataset__Average_for_2013-2015.csv") as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		if (str(row[0]) == "Otsego, Schoharie, Oneida, & Herkimer"):
			county_income.append(("Otsego" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Schoharie" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Oneida" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Schoharie" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Herkimer" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Broome, Chenango, Delaware, & Tioga"):
			county_income.append(("Broome" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Chenango" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Delaware" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Tioga" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Clinton, Franklin, Essex & Hamilton"):
			county_income.append(("Clinton" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Franklin" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Essex" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Hamilton" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Steuben, Schuyler & Chemung"):
			county_income.append(("Steuben" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Schuyler" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Chemung" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Cattaraugus & Allegany"):
			county_income.append(("Cattaraugus" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Allegany" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Ontario & Yates"):
			county_income.append(("Ontario" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Yates" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Warren & Washington"):
			county_income.append(("Warren" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Washington" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Livingston & Wyoming"):
			county_income.append(("Livingston" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Wyoming" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Genesee & Orleans"):
			county_income.append(("Genesee" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Orleans" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Sullivan & Ulster"):
			county_income.append(("Sullivan" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Ulster" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Fulton & Montgomery"):
			county_income.append(("Fulton" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Montgomery" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Wayne & Seneca"):
			county_income.append(("Wayne" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Seneca" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
		elif (str(row[0]) == "Jefferson & Lewis"):
			county_income.append(("Jefferson" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Lewis" + " County"))
			income_range.append((row[4]))
			income_median.append(0)	
		elif (str(row[0]) == "Columbia & Greene"):
			county_income.append(("Columbia" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Greene" + " County"))
			income_range.append((row[4]))
			income_median.append(0)	
		elif (str(row[0]) == "Madison & Cortland"):
			county_income.append(("Madison" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Cortland" + " County"))
			income_range.append((row[4]))
			income_median.append(0)	
		elif (str(row[0]) == "Cayuga & Onondaga"):
			county_income.append(("Cayuga" + " County"))
			income_range.append((row[4]))
			income_median.append(0)
			county_income.append(("Onondaga" + " County"))
			income_range.append((row[4]))
			income_median.append(0)													
		else: 
			county_income.append((str(row[0]) + " County"))
			income_range.append((row[4]))
			income_median.append(0)



for x in range(len(income_range)):
	if (income_range[x] == "$0 to <$10,000"):
		income_median[x] = 5000
	elif (income_range[x] == "$10,000-<$20,000"):
		income_median[x] = 15000
	elif (income_range[x] == "$20,000-<$30,000"):
		income_median[x] = 25000
	elif (income_range[x] == "$30,000-<$40,000"):
		income_median[x] = 35000
	elif (income_range[x] == "$40,000-<$50,000"):
		income_median[x] = 45000
	elif (income_range[x] == "$50,000+"):
		income_median[x] = 55000


with open("county_median_income3.csv","w+") as csv_file:  
    for x in range(len(county_income)):
        row = str(county_income[x]) +","+ str(income_median[x])
        csv_file.write(row+'\n')



###########################################################################
# Heart Diesease dataset
county_heart = []
heart_disease = [] # per 100000
# only saving those counties which are common to both datasets
with open("Heart_Disease_Mortality_Data_Among_US_Adults__35___by_State_Territory_and_County.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
    	if not row[7]:
    		continue
    	state_name = str(row[1])
    	county_name = str(row[2]) 
    	if (state_name == "NY"):
    		county_heart.append(county_name)
    		heart_disease.append(float(row[7]))


with open("county_heart_disease3.csv","w+") as csv_file:  
    for x in range(len(county_heart)):
        row = str(county_heart[x]) +","+ str(heart_disease[x])
        csv_file.write(row+'\n')