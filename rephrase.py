import torch
from sentence_transformers import SentenceTransformer, util

import os
from typing import Union
import tqdm

import pickle
import glob
import json

import gc


gguf_base_path = "/opt/IdExtend/models/llm/"
gguf_models = [
    "phi-4-Q5_K_M.gguf",
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf",
    "Ministral-8B-Instruct-2410.Q5_K_M.gguf",
    "Qwen2.5-14B-Instruct-Q5_K_L.gguf",
    "Mistral-Small-24B-Instruct-2501-Q6_K.gguf",
]



if __name__ == "__main__":
    model_embeddings = SentenceTransformer(
        "Alibaba-NLP/gte-modernbert-base",
        model_kwargs={"torch_dtype": torch.float16},
        device="cuda:2",
        trust_remote_code=True,
    )

pickl_file_cache = "cached_chat_per_llm.pkl"


if os.path.exists(pickl_file_cache):
    with open(pickl_file_cache, "rb") as f:
        cached_prompts_per_llm = pickle.load(f)
else:
    cached_prompts_per_llm = {}

prompts = []

cur_file_location = os.path.dirname(os.path.realpath(__file__))
prompts_path = os.path.join(cur_file_location, "prompts")
_glob_prompts = glob.glob(prompts_path + "/*.txt")
#sort by name casted to int
_glob_prompts.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
for file in _glob_prompts:
    with open(file, "r", encoding="utf-8-sig") as f:
        prompts.append(f.read())

_glob_contexts = glob.glob("./prompts_tester/contexts/*.json")
#sort by name casted to int
_glob_contexts.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
contextes = []
for file in _glob_contexts:
    with open(file, "r", encoding="utf-8-sig") as f:
        contextes.append(json.load(f))

def index_to_letter(index):
    return chr(index + 65)




total_scores = {}
# from vllm import LLM, SamplingParams
from llama_cpp import Llama
total_question = sum([len(context["questions"]) for context in contextes])
for vllm_model_name in gguf_models:
    if not vllm_model_name in cached_prompts_per_llm:
        cached_prompts_per_llm[vllm_model_name] = {}

    scores = {}
    batches = []
    for prompt_index, prompt in enumerate(prompts):
        total_score = 0

        if not prompt in cached_prompts_per_llm[vllm_model_name]:
            cached_prompts_per_llm[vllm_model_name][prompt] = {}

        for context in contextes:
            turns:list[dict] = context["turns"]
            testing_questions:list[dict] = context["questions"]

            for dict_question in testing_questions:

                question, expected_string = dict_question['question'], dict_question['expected']
                required_text:Union[str, list[str]] = dict_question.get('required_text', None)

                prompt_formated = prompt.format(question=question)

                tmp_turns = turns.copy()

                tmp_turns.append({"role": "user", "content": prompt_formated})
                batches.append((prompt_index, tmp_turns, question, expected_string, required_text, context['context_name']))


    cmp_mode = 1

    llm = None

    # llm = LLM(vllm_model_name, trust_remote_code=True, max_model_len=2048)
    """try:
        llm = Llama(
            gguf_base_path + vllm_model_name,
            logits_all=False,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
            flash_attn=True,
        )
    except Exception as e:
        print(e)
        continue"""
    #lazy load beacause of cached prompts

    answers = []
    # outputs = llm.chat([chat_completion[1] for chat_completion in batches], sampling_params)
    for chat_completion in tqdm.tqdm(batches):
        # if vllm_model_name in cached_prompts_per_llm and chat_completion[2] in cached_prompts_per_llm[vllm_model_name]:
        #    answers.append(cached_prompts_per_llm[vllm_model_name][chat_completion[2]])
        #    continue

        if (
            vllm_model_name in cached_prompts_per_llm
            and prompts[chat_completion[0]] in cached_prompts_per_llm[vllm_model_name]
            and chat_completion[2]
            in cached_prompts_per_llm[vllm_model_name][prompts[chat_completion[0]]]
        ):
            answers.append(
                cached_prompts_per_llm[vllm_model_name][prompts[chat_completion[0]]][
                    chat_completion[2]
                ]
            )
            continue


        if llm is None:
            try:
                llm = Llama(
                    gguf_base_path + vllm_model_name,
                    logits_all=False,
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    verbose=False,
                    flash_attn=True,
                )
            except Exception as e:
                print('Could not load the gguf model, got error ', str(e))
                llm = False
                break

        result = llm.create_chat_completion(chat_completion[1], temperature=0.7)
        answers.append(result["choices"][0]["message"]["content"])
    if llm is False:
        continue
    scores_per_id = [0] * len(prompts)

    for answer_index, answer in enumerate(answers):


        prompt_index, tmp_turns, question, expected_string, required_text, context_name = batches[answer_index]

        cached_prompts_per_llm[vllm_model_name][prompts[prompt_index]][
            question
        ] = answer

        # output_str = output['choices'][0]['message']['content']
        output_str = answer

        worse_possible_output = model_embeddings.encode([question])
        perfect_embeddings_expected = model_embeddings.encode([expected_string])
        ground_truth_output = model_embeddings.encode([output_str])

        sim_perfect_worse = util.pytorch_cos_sim(
            perfect_embeddings_expected, worse_possible_output
        ).item()

        sim_ground_worse = util.pytorch_cos_sim(
            ground_truth_output, worse_possible_output
        ).item()
        sim_ground_perfect = util.pytorch_cos_sim(
            ground_truth_output, perfect_embeddings_expected
        ).item()

        prune_reason = None
        if cmp_mode == 1:

            if len(output_str) > (len(expected_string) * 2):
                score = 0
                prune_reason = "TOO LONG"

            if sim_ground_perfect > sim_ground_worse:
                if sim_ground_perfect > 0.86:
                    score = 1
                else:
                    score = 0.0
                    prune_reason = "TOO LOW SIMILARITY"
            else:
                score = 0
                prune_reason = "TOO CLOSE TO WORSE POSSIBLE OUTPUT"
        elif cmp_mode == 2:
            score = sim_ground_perfect
            
        if required_text is not None:
            if isinstance(required_text, str):
                if required_text.lower() not in output_str.lower():
                    score = 0
                    prune_reason = "REQUIRED TEXT NOT FOUND"
            elif isinstance(required_text, list):
                if not all([txt.lower() in output_str.lower() for txt in required_text]):
                    score = 0
                    prune_reason = "REQUIRED TEXTS NOT FOUND"
                    
        if "\n" in output_str.strip():
            score = 0
            prune_reason = "NEWLINE DETECTED WHICH PROBABLY MEANS EXTRA COMMENT"

        # take highest cosin similarity

        print(
            f"\t {index_to_letter(prompt_index)})  Question :  {question}   Answer : {output_str} {score} -> {sim_ground_perfect} {prune_reason if prune_reason is not None else ''}"
        )

        scores_per_id[prompt_index] += score

    print("Model :", vllm_model_name)
    for prompt_index, score in enumerate(scores_per_id):
        print(f"Prompt {index_to_letter(prompt_index)} : {score:.2f}")
        scores_per_id[prompt_index] = score / total_question

    total_scores[vllm_model_name] = scores_per_id
    if llm:
        del llm
    gc.collect()
    torch.cuda.empty_cache()


pickle.dump(cached_prompts_per_llm, open(pickl_file_cache, "wb"))
with open("scores.json", "w") as f:
    json.dump(total_scores, f)

print('Total questions :', total_question)