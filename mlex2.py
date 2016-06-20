# PRACTICE 2 SETUP


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
raw = fetch_20newsgroups(subset='all', remove= ('headers', 'footers', 'quotes'))
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import time
#start_time = time.time()
#main()
#print("--- %s seconds ---" % (time.time() - start_time))

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(raw.data)

vectors.shape # Q02 how many patterns? what is the input-dimensionality? what is the input space?
vlen = vectors.shape[0]
vwid = vectors.shape[1]
vals =(vlen*vwid)
print("vec shape", vectors.shape)
# The input dimensionality is 18,846 by 134,410

vectors.nnz # Q03 what is the % of non-zero values in the matrix?

print("vec non-zeros", vectors.nnz)
print("vec values", vals)
print("vec zeros", vals - vectors.nnz)
print("percent zeros", (vals - vectors.nnz)/vals)

# Over 99.9% of the values are zeros.
# This only makes sense because there are nearly 10x as many features as documents
# and the vast majority of these features don't occur in any give document.
# The average document has fewer than 1 in 1000 features (or 134 in 134,000).

print("vec len",vlen)
# The document has 18,846 observations.
vlen*.2


# run vs. last 5000
#testsize = vlen*.25
#thresh = vlen - testsize
training_size = []
test_accuracy = []

for ts in range(1,18001,10):
	clf = MultinomialNB(alpha=.01)
	clf.fit(vectors[0:ts,], raw.target[0:ts])
	pred = clf.predict(vectors[18001:,])
	met = metrics.f1_score(raw.target[18001:], pred, average='weighted')
	training_size.append(ts)
	test_accuracy.append(met)
	#print(met)

print(len(test_accuracy))
print(len(training_size))
plt.plot(training_size,test_accuracy)
plt.show()


'''
#Example run: 

ts=5000

clf = MultinomialNB(alpha=.01)
clf.fit(vectors[0:ts,], raw.target[0:ts])
pred = clf.predict(vectors[ts+1:,])
met = metrics.f1_score(raw.target[ts+1:], pred, average='weighted')

#print(met)
'''