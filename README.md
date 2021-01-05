# **MATHEMATICS OF DEEP LEARNING ALGORITHMS 2 - Final Project**
# ***TEXT GENERATION: Algorithm for sign-specific horoscopes generation***
## **PART 1: Data Preprocessing**
### **Step 1: Dataset import**

The dataset used is composed of the **daily horoscopes from 2013 to 2016** published on the NY Post website. You can find it at the following link: 

https://github.com/dsnam/markovscope/tree/master/data?fbclid=IwAR38nDSVK3GPwQbHf864wMEbR0oawBEGBy5QWPlGsmUI60orKTJVzFTmFwo

**Description of the Dataset**

The dataset is under CSV format and records **4 different variables**: 


*   a variable which associates a unique index number to each observation, that we rename **id**;

*   a variable which stores the full text horoscope for this observation, called **horoscope**;

*   a date variable corresponding to the publication date of each observation, named **date**

*   a variable indicating the sign to which the daily horoscope is associated, renamed **sign**.

The Horoscope dataset contains 12962 observations, the daily horoscopes.
Example: horoscope for aries on the 12 january 2013: 


"You’re not the sort to play safe and even if you have been a bit more cautious than usual in recent weeks you will more than make up for it over the next few days. Plan your new adventure today and start working on it tomorrow."

**Number and Proportion of horoscope relative to each sign**

To carry out our project, we look to see if all the signs are well represented in our dataset. We obtain the following result:

TABLEAU

Hence, each sign is represented by on average 1080 horoscope examples. 
The small differences between signs suggest that some sign's horoscopes were not scrapped from the NY post website on several dates. 
This is not an issue, since we have enough examples for each sign to perform analysis. 

### **Step 2: Cleaning step for NLP methods**

**Cleaning procedure of the data**

We create several function to clean our dataset:

*  to convert all the text to lowercase ;
*  to remove punctuation ;
*  to tokenize the sentences;
*  to remove the general stop words of english;
*  to remove the specific stop words of horoscope;
*  to reverse the tokenization process;

With these functions, we define three functions to have our final dataset:

*  the first one covert to lowercase,remove the punctuation, tokenize the sentence, remove the stop words and finally reverse the tokenization process ;
*  the second one is the same than the first but we add the removal of specific stop words ;
*  the last one is the same than the first but we do not remove the stop words ;

### **Step 3: By sign words analysis and wordcloud representation**

The goal is to display under a illustrative form the most relevant words for each sign, in order to highlight whether the horoscopes are sign-specific. 

We aim at generating horoscope according to the choosen sign, which makes sense if each sign is characterized prediction consistence with its supposed personnality traits. 

We will propose two analysis ways: 

*   **"Most frequent words" analysis** which basically consist in highlighting the most used words for each sign mamong all the available horoscope prediction examples.
This method requires to use the version of cleaned predictions which does not include the horoscope specific stopword.

*   **TF-IDF analysis** which highlights the most used representative words for a sign, by taking into account both the frequency of each word in the sign corpus and the specificity of these words to this sign corpus comparing to other signs ones. 
Hence for this method we used the horoscope predictions set without removing the horoscope specific stopwords because they should not be consider as important words since they appears in all sign corpus.


We use the second cleaning function to do these study.


**Method 1: Most frequent words**

The first method gives us the following wordcloud (for the 20 more frequent word):
Examples for ....
IMAGE

This direct method does not yield satisfying results: most frequent words among the horoscope of each sign are similar and do not give explicit clue about sign's specificities.


**Method 2: TF-IDF analysis**

We display the TF-IDF wordcloud associated to EXAMPLE of the list of the 20 words with higher TF-IDF scores.

This method produces interesting results since each signs is associated with a different set of words that we expect to be related to the personality of this sign.

Here is some TF-IDF comparison with sign personnality traits according to the websites: 
http://nuclear.ucdavis.edu/~rpicha/personal/astrology/

https://blog.prepscholar.com/cancer-traits-personality 

https://blog.prepscholar.com/taurus-traits-personality 

We choose arbitrarly our two zodiac signs for comparison: **cancer** and **taurus**

*   **Cancer**

Expected characteristics of the cancer: 

    - Emotional & loving & loyal & caring & generous
    - Intuitive & imaginative
    - Shrewd & cautious
    - Protective & sympathetic
    - Changeable & moody & enigmatic
    - Overemotional & touchy
    - Clinging & unable to let go & vindictive

Obtained characteristics of the cancer: 

    - generosity & giving & receiving & someone
    - mind & recharge & nature
    - matter & mind & resolute & decision
    - break & change
    - resolute & decision
    
    
 This comparison seems quite accurate.
 
 *   **Taurus**
 
 Expected characteristics of the Taurus: 

    - Patient & reliable & value honesty above all else
    - Warmhearted & loving
    - Persistent & determined & hard-worker & ambitious
    - Security loving & placid & confort in stability and consistency
    - Can take their pleasure-seeking ways too far
    - Resentful & inflexible & perfectionists
    - Greedy & enjoy things luxurious and cozy

Obtained characteristics of the Taurus: 

    - relationship & tells & clearly & relationship & friend person
    - mind & recharge & nature
    - work & making & plunge & better & far
    - fear & suspicious
    - plunge & better & far
    - thing
 
This comparison seems also quite accurate. 
 
 **Conclusion Step 3**
 
 The horoscope predictions from our dataset seems to be at least partially sign-specifics since the TF-IDF methods (which also takes into account that a word is rare among the other text of the corpus) highlights different words for each sign. Moreover these words seems to be linked to the personality traits of each sign. 

Hence it makes sense to building a text generator algorithm able to take into account the sign to produce the horoscope. 

## **PART 2: Horoscope predictions generation**

### **Step 1 : Create train and test sets**
 
 For our project, we split the original dataset into a train and a test set. The first one correpond to 90 % of the data and the second is 10% of our data.
 
 Once again, we check the composition of our train and test sets.
 
 **Train set composition**
 
 TABLEAU
 
 **Test set composition**
 
 TABLEAU
 
 Our datasets have a god composition and all the signs are well represented.
 We use the third cleaning function to have logical sentences.
 
 ### **Step 2: Model specification and associated processing methods**
 
 Our text generation model relies on several functions: 

**Function *get_sequence_of_tokens***

Transformation of original sentences into N-Grams tokens

**Function *generate_padded_sequences***

Transformation of N-Grams token into padded vectors (predictors and label, i.e. the next word of the initial sentence).
The set of tokens sequences might include sequences of different lengths. Before starting training the model, we need to pad the sequences and make their lengths equal. 
The function creted an input vector, which contains same length N-grams sequence build from the corpus, and a label vector, which is the next word of the N-gram, which can be used for the training purposes.

**Function *prep_LSTM***

Select the subset of the initial horoscope predictions associated to the targetter sign (passed as an argument) and apply to the resulting list of sentences the two preprocessing functions mentionned above.
This function creates the N-grams sequences and apply the padding process to the subset of horoscope predictions associated to the targetted sign.

**Function *create_model***

Generation of the LSTM model, this model is created according to: 
* the maximum length of padded vectors generated from the training set,
* the total number of words in the vocabulary used in the training corpus. 

The model used is a Long short-term memory (LSTM) model, i.e. is build as an artificial recurrent neural network (RNN) architecture.
It includes: 


*   An *embedding* input layer 
*   Two LSTM *hidden layers* with *dropout rate* of 10%
*   One *hidden layer* with *relu activation function*
*   An *output layer* with a *softmax activation function*

The optimizer used is *ADAM* and the loss minimized is the *categorical cross-entropy*. 

**Function *generate_text*** 

Generate new sentences containing a requested number of words, from a initial text extract.
This function takes an initial horoscope prediction (from the initial dataset, in generak from a test set independent from the training set).
It returns the initial horoscope predictions extended by a certain number (controlled by the argument *next_words*) of new predicted words which form the artificially generated horoscope. 

### **Step 3: Test of the text generation model**

This step consists in implemented the above model for an arbitrarly choosen sign to check the execution and evaluate the results of the generative functions introduced previously. 

If the model is consider satisfying, the last step would be to implement a global horoscope generator function. 
 
We can display an example for the sign Aries, we choose to generate 100 additional words. We have the following result:   
 
**Original sentence :**

You Are Wasting Too Much Time And Too Much Energy On Things Of No Importance  The Sun S Move Into The Work Area Of Your Chart This Weekend Makes This The Perfect Day To Start Reorganizing Your Life  Get Rid Of Everything That Isn T Essential

**Prediction:**

That You Are Doing You Will Be Able To Explain What You Are Doing You Will Be Doing You Will Pays Alongside Alike Are Anywhere Games Perfectly Deals You Arrives Begs Alongside Is Not A Lot Alongside Calmly In The Moment But Wherever You Have To Do It Is A Danger And Therefore Be A Bit Of The Most Dynamic Area Of Your Chart Today You Will Be A Bit Of The Most Dynamic Area Of Your Chart Today You Will Be A Bit Alongside Alike Pays On The Past And Putting Your Way And Energy On You Will Be


This result is quite good and we can find some words corresponding to horscope lexical like 'Danger','Past','Energy' and 'Camly'. Most of the grammar is wrong and we can't really distinguish a sentence. The end of prediction seems to be a repetition of the previous words.


### **Step 4: Global horoscope prediction function**

**Global function 1**

With this function, we just have to choose the sign we want to predict the horoscope for and the number of word we want to generate.

Here an example of output with the sign cancer and 100 words generated:

**Original sentence :**

What You Expect To Happen Today And Over The Weekend Most Likely Won T  While What You Don T Expect To Happen Most Likely Will  In Which Case There Is No Point Making Plans As They Are Certain To Change  Take Life As It Comes 

**Prediction:**

To The Wind You Will Get It Easy To Get To The End Of The That You Will Be Amazed If You Are Smart You To To You Regret A Few A Direction That Is You To To Get To What To To You To To Do A Idea On To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You To To You

This result is not as good as the first one but we can still find some words corresponding to horscope lexical like 'Direction' and 'Regret'. Once again the grammar is wrong and we can't really distinguish a sentence. We find the repetition of the sames words at the end of the prediction, even more than previously.

**Global function(s) 2: One can also prefer using 2 separate functions: one to build and train the model and one to predict.**

We create two new functions to re-use a trained model if we want to make several predictions for the same size and sign. 

It is particularly usefull since the computational time required for the model to be trained is quite long.

The first function build and train the model. The second one generate and display the prediction.

Here an example of result for the sign Aries and 100 generated words:
