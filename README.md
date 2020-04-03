# Natural-Language-Processing-Project
This is a repository on my NLP project on sentiment analysis using Twitter Data 

You can access the data and the problem statement from here: https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

### PROBLEM STATEMENT : Classifiy the given set of tweets into Non Racial (class 0) and Racial (class 1) tweets.

### Sample Data Set:

![Data Sample](https://user-images.githubusercontent.com/53376072/78349979-0a971480-75c2-11ea-858e-f0742bef13b1.JPG)

As you can see, the data set requires a lot of cleaning before feeding into the models.
This is the reason that extensive use of regular expression was done for cleaning puporses.


<font size ='4'>**Libraries used**</font>
* Numpy
* Pandas
* Seaborn (visualization)
* Matplotlib (visualization)
* Regular Expressions (Data Cleaning)
* Wordcloud (visualization)
* NLTK (Text analysis and processing)
* SPACY (Text analysis and processing)
* Sklearn (Model building and deployment)
* Genism ( Model building - Word vectors using Word2vec )

<font size ='4'>**Work flow of the Data**</font>

Below is the sample of data after performingcleaning and data processing steps like removing stopwords,stemming and lemmatization. I have tried to compare the functioning of **NLTK** and **Spacy** libraries.

![Data after](https://user-images.githubusercontent.com/53376072/78354445-7c269100-75c9-11ea-8ed0-471b992b5797.JPG)



## VISUALIZATION

We will perform visualizations using a wordCloud. WordCloud basically displays the most commonly occuring words of a document.
The bigger the size of the word in the wordcloud, the more the occurence of that word in the document.
<br>
</br>
<font size = '3'>Hence its a quick way to know about the important words occuring in the tweets.</font>

<font size ='4'>**(1.) Analysis of Non-Racist Tweets**</font>

![Analysis of non racist tweets](https://user-images.githubusercontent.com/53376072/78353856-6a90b980-75c8-11ea-9752-8219241b8552.JPG)

We can clearly see the words like happy, love, smile and thank you occuring regularly in non racial tweets.


<font size ='4'>**(2.) Analysis of Racist Tweets**</font>

![Analysis of racist tweets](https://user-images.githubusercontent.com/53376072/78354117-ebe84c00-75c8-11ea-96ae-996d09558ee1.JPG)

<font size ='4'>**(3.) Analysis of Most Common Hashtags in Non Racist Tweets**</font>

![Non Racial hashtags](https://user-images.githubusercontent.com/53376072/78355466-58644a80-75cb-11ea-9e79-1788d2b6612e.JPG)


<font size ='4'>**(4.) Analysis of Most Common Hashtags in Racist Tweets**</font>

![Racial hashtags](https://user-images.githubusercontent.com/53376072/78355625-a0836d00-75cb-11ea-9d9f-ce4bf933ef36.JPG)


<font size ='4'>**(5.) Visualization of Named Entity Recogonition.**</font>

Tags are accessible through the `.label_` property of an entity.
<table>
<tr><th>TYPE</th><th>DESCRIPTION</th><th>EXAMPLE</th></tr>
<tr><td>`PERSON`</td><td>People, including fictional.</td><td>*Fred Flintstone*</td></tr>
<tr><td>`NORP`</td><td>Nationalities or religious or political groups.</td><td>*The Republican Party*</td></tr>
<tr><td>`FAC`</td><td>Buildings, airports, highways, bridges, etc.</td><td>*Logan International Airport, The Golden Gate*</td></tr>
<tr><td>`ORG`</td><td>Companies, agencies, institutions, etc.</td><td>*Microsoft, FBI, MIT*</td></tr>
<tr><td>`GPE`</td><td>Countries, cities, states.</td><td>*France, UAR, Chicago, Idaho*</td></tr>
<tr><td>`LOC`</td><td>Non-GPE locations, mountain ranges, bodies of water.</td><td>*Europe, Nile River, Midwest*</td></tr>
<tr><td>`PRODUCT`</td><td>Objects, vehicles, foods, etc. (Not services.)</td><td>*Formula 1*</td></tr>
<tr><td>`EVENT`</td><td>Named hurricanes, battles, wars, sports events, etc.</td><td>*Olympic Games*</td></tr>
<tr><td>`WORK_OF_ART`</td><td>Titles of books, songs, etc.</td><td>*The Mona Lisa*</td></tr>
<tr><td>`LAW`</td><td>Named documents made into laws.</td><td>*Roe v. Wade*</td></tr>
<tr><td>`LANGUAGE`</td><td>Any named language.</td><td>*English*</td></tr>
<tr><td>`DATE`</td><td>Absolute or relative dates or periods.</td><td>*20 July 1969*</td></tr>
<tr><td>`TIME`</td><td>Times smaller than a day.</td><td>*Four hours*</td></tr>
<tr><td>`PERCENT`</td><td>Percentage, including "%".</td><td>*Eighty percent*</td></tr>
<tr><td>`MONEY`</td><td>Monetary values, including unit.</td><td>*Twenty Cents*</td></tr>
<tr><td>`QUANTITY`</td><td>Measurements, as of weight or distance.</td><td>*Several kilometers, 55kg*</td></tr>
<tr><td>`ORDINAL`</td><td>"first", "second", etc.</td><td>*9th, Ninth*</td></tr>
<tr><td>`CARDINAL`</td><td>Numerals that do not fall under another type.</td><td>*2, Two, Fifty-two*</td></tr>
</table>

<font size ='4'>**Function used for NER visualization**</font>

list_1 = []

named_entity_recogonition =  all_data['tweet_lemma_spacy'].loc[0:40000].apply(lambda x: [ent.text for ent in nlp(x).ents if (ent.label_ == 'PERSON' or ent.label_ == 'NORP' )])

for i in named_entity_recogonition:
        list_1.extend(i)
        
persons_mentioned = ' '.join(list_1)

![NER](https://user-images.githubusercontent.com/53376072/78356394-1b00bc80-75cd-11ea-93c5-d321aadfe47e.JPG)
