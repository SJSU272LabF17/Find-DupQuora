# Project-Team-17

## Title
Detecting duplicate questions on Quora

## Objective
This project aims to find the duplicate questions on Quora using NLP and Data Mining Techniques.

## Team Members
1) Thejaswi Kampalli
2) Manogna Sindhusha Mujje
3) Pooja Shivasharanappa
4) Harsh Patel

## About Project:
We started off the project learning and discussing the basic concepts of Data Mining and Natural language Processing. After some research on the implementation of the project, the solution is boiled down to cosine similarity calculation using term frequencies. This program implements cosine similarity technique by building CSR Matrix which contains term frequencies of each question as a vector. There is still a lot of scope for improving the accuracy by using some Neural Networks concepts and Deep Learning embedding techniques.

the techniques involves calculating the cosine similarity between questions in the pair by frequency scores in the CSR Matrix built. 

## Dataset:
Quora has released a dataset consisting of question pairs earlier this year. The dataset can be downloaded from this link http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv It includes 404351 question pairs with a label column indicating if they are duplicate or not. The score listed for each question pair as 1 if both are simalar and 0 if both are different. The dataset has 63.07% labelled as non duplicates and 36.93% labelled as duplicates. 

## Note: 
The dataset file is too big to run the cosine similarity technique in our machines, hence we executed the code with a smaller dataset on a Jupyter Notebook.
