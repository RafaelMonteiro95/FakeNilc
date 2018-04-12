import preprocess.reduce as rc
import os

if __name__ == '__main__':
	
	news_dir = 'db'
	output_dir = 'var/texts'

	# creating dir for storing reduced texts
	os.makedirs(output_dir, exist_ok=True)
	output_dir += '/'
	os.makedirs(output_dir + 'real', exist_ok=True)
	os.makedirs(output_dir + 'fake', exist_ok=True)
	

	#fetching files
	filenames = []
	for real, fake in zip(os.listdir(news_dir + '/real'),os.listdir(news_dir + '/fake')):
		#appends a tuple with a true and a fake file filename
		filenames.append((news_dir + '/real/' + real, news_dir + '/fake/' + fake))

	# reducing files lenght
	for pair in filenames:

		real_filename = pair[0]
		fake_filename = pair[1]
		real_name = real_filename.split('/')[-1]
		fake_name = fake_filename.split('/')[-1]

		#opening files
		with open(pair[0], encoding='utf8') as real:
			with open(pair[1], encoding='utf8') as fake:

				#read both files and reduce the lenght of the biggest
				result = rc.reduce(real.read(),fake.read())

				with open(output_dir + 'real/' + real_name,'w', encoding='utf8') as f:
					print(result[0],file=f)
				with open(output_dir + 'fake/' + fake_name,'w', encoding='utf8') as f:
					print(result[1],file=f)