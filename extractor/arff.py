def createARFF(attributes, data, relation, filename):

	with open(filename+'.arff','w', encoding='utf8') as f:
		# relation name
		print('@relation {0}\n'.format(relation) ,file=f)

		#each attribute name and value:
		for att in attributes:
			if(att == 'Trustworthy'):
				print('@attribute',att,'{TRUE, FALSE}',file=f)
			else:
				print('@attribute',att,'numeric',file=f)

		print('\n@data\n',file=f)

		# for each instance in the dataset
		for instance in data:
			# attributes of this instance
			print(*instance, sep=', ', file=f)


def createCSV(attributes, data, relation, filename, tags = None):

	with open(filename + '.csv','w', encoding='utf8') as f:

		#print labels
		print(*attributes, sep=',',end = '', file=f)

		if(relation == "Trustworthy"):
			print(file=f)
			for i in range(len(data)):
				print(data[i],file=f)

		else :

			#printing last label or line end
			if(tags != None):
				print(',Trustworthy', file=f)
			else:
				print(file=f)

			#print each instance
			for i in range(len(data)):
				#if im printing labeled data, prints the label in the end
				if(tags != None):
					print(','+tags[i],sep='', file=f)
				#else, justs prints the data
				else:
					print(*data[i], sep=',', file=f)

	return filename + '.csv'