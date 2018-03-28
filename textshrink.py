import os

def shrinkTexts():

	print('Creating inputs directory...',end='')
	os.makedirs('input', exist_ok=True)
	os.makedirs('input/false', exist_ok=True)
	os.makedirs('input/true', exist_ok=True)
	print('Done!')

	# filenames = ['100.txt']

	filenames = os.listdir('data/true')
	for filename in filenames:
		print('Processing file ',filename,'...',flush=True,end='',sep='')

		#counting number of words
		with open('data/false/' + filename, encoding='utf8') as f:
			fake_text = f.read()

		with open('data/true/' + filename, encoding='utf8') as f:
			true_text = f.read()

		fake_count = 0
		for word in fake_text.split():
			fake_count += 1

		true_count = 0
		for word in true_text.split():
			true_count += 1

		#selecting which text will be shrinked
		bigger_text = true_text if (true_count > fake_count) else fake_text
		# bigger_text = true_text

		#shrinking text
		sentences = [] #list that keeps the sentences that will stay in the text
		word_count = 0 #auxiliary word count

		#for each sentence in the big text
		for sentence in bigger_text.splitlines():

			if(sentence.rstrip() == ''):
				continue

			#counts words in that sentece
			for word in sentence.split():
				word_count += 1

			#if i've included enough words, ill stop including sentences
			if word_count > fake_count:
				break

			sentences.append(sentence)

		# print(sentences)
		#joins sentences into a resulting text
		resulting_true_text = '\n'.join(sentences) if (true_count > fake_count) else true_text
		resulting_false_text = '\n'.join(sentences) if (true_count < fake_count) else fake_text

		with open('input/true/' + filename, 'w', encoding='utf8') as f:
			print(resulting_true_text, file=f)
		with open('input/false/' + filename, 'w', encoding='utf8') as f:
			print(resulting_false_text, file=f)
		print('Done!',flush=True)