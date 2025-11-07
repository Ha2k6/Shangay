import re, tensorflow as tf

from tensorflow.keras.datasets import imdb



(x,_),_ = imdb.load_data(num_words=5000)

w = imdb.get_word_index(); inv = {i+3: w for w,i in w.items()}



STOP = set("the a an and is it to of this that i you was for in on with as are be".split())

dec = lambda s: " ".join(inv.get(i,"") for i in s)

clean = lambda t: " ".join([w for w in re.sub(r"[^a-zA-Z ]"," ",re.sub(r"<.*?>"," ",t)).split() if len(w)>1 and w not in STOP])



print(clean(dec(x[0]))[:200])

