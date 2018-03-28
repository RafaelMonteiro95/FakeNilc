def createARFF(attributes, data, relation, filename):

	with open(filename,'w', encoding='utf8') as f:
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
 