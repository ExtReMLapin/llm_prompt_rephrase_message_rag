# llm_prompt_rephrase_message_rag
Quick tool to kinda permute/bruteforce few combinaisons of model/prompts/contexts to evaluate each and see if the llm can rephrase the last user message to be used in a RAG context


Long story short, in this chatbox RAG context : 


>Q : What is Sony ?
>
>A : It's a company working in tech.
>
>Q : How much money did they make last year ?

If you're working with a RAG (embeddings+(optional reranker), you pretty much can't do anything with the sentence `How much money did they make last year ?` because your embeddings DB will not know what/who is `they`

The idea is to use a llm to rephrase this sentence using the chat history.

A LLM out of a RAG context can perfectly use this answer, but here we want to use it in a RAG context.


In this repo you'll find the folder `contexts` containing handmade very short chat histories and questions/expected answer that will be used together with `prompts` folder.

`prompts` folder contains a series of prompts that performs good or bad depending on the model.

Prompts are all in english.

Half of the contexts (and their questions) are in French, other half in English.


### Installation

You'll need python llama cpp but to be honest you can also easily reimplement the llm call to any other tool.

Just run the python files and read the packages errors


### Usage

use rephrase.py to run the bruteforce thing benchmark, it will generate a `score.json` file then use `excell.py` to generate a `model_scores_comparison_3.xlsx` file.

You can run it multiple times, prompts are cached in a pkl file.

xlsx file looks like this (you'll have to make your own graph)


![image](https://github.com/user-attachments/assets/cb331134-d435-4cee-a376-0a7e5479e420)



## **FEEL FREE TO HELP ME ADDING PROMPTS AND CONTEXTS WITH EASY OR HARDS CHALLENGES FOR THE LLM**
