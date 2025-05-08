from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
import re
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import numpy as np
from tqdm import tqdm
from eval_utils import compute_scores
import torch
from torch.optim import AdamW
import json
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from trl.core import LengthSampler
import pandas as pd
# from evaluate import evaluate
from utils import (
    prepare_args,
    prepare_data,
    load_pretrained,
    preprocess_data,
    PPODataCollatorForChatGLM,
    PPOTrainerForChatGLM,
    compute_rewards,
    get_logits_processor,
    plot_loss
)
# device_ids = [0, 1] # 可用GPU

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a T5 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config` then run the script with
# `accelerate launch ppo-sentiment-t5-small.py`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="./output_dir_pt_20_single/global_step-7200/", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )

outputss, targets,sample3 = [], [],[]
def evaluate(pmodel,model,tokenizer,gen_kwargs,dataset):
    f1 = 0.0
    max_tgt_len = 768 - 450 - 3
    save_data = []
    pmodel.gradient_checkpointing_disable()
    pmodel.config.use_cache = True
    pmodel = torch.nn.DataParallel(pmodel).cuda()
    with open('./data/test_black.json', "r", encoding="utf-8") as fh:
        sample2 = json.load(fh)
        num=0
        for i in sample2:
            num+=1
            print(num)
            with torch.no_grad():
                sample = i
                src_tokens = tokenizer.tokenize(sample["text"])
                sample3.append(sample["text"].split(' '))
                prompt_tokens = tokenizer.tokenize('')

                if len(src_tokens) > 768 - len(prompt_tokens):
                    src_tokens = src_tokens[:768 - len(prompt_tokens)]

                tokens =  src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_ids = tokenizer.encode("帮我写个快排算法")

                input_ids = torch.tensor([input_ids]).to("cuda:0")
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                    "num_beams":8,
                    "return_dict_in_generate":True
                }
                response = model.generate(input_ids,max_length=768, **gen_kwargs)
                res = []
                response=response.sequences
                outputs = response[0][input_ids.shape[1]:]
             
                r = tokenizer.decode(outputs).replace("<eop>", "")
             
                r = r.split('文本抽取的信息如下:')[1]
               
                res.append(r)
                # print('predict:',res[0].replace('\n',''))
                # print('answer:',sample['answer'])
                outputss.append(res[0])
                targets.append(sample['answer'])
 
        raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputss, targets,sample3 )
        results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
                        'preds': all_preds, 'preds_fixed': all_preds_fixed}
        exp_results = f"Raw F1 = {raw_scores['f1']:.4f}, Fixed F1 = {fixed_scores['f1']:.4f}"
        print(exp_results)
        pmodel.gradient_checkpointing_enable()
        pmodel.config.use_cache = False

def compute_genscore(text,label):
    right=0.0
    total=0.0
    text = text.split('\n')
    label = label[0].split('\n')
    textdict={}
    labeldict={}
    for pt in text:
        
        pt = pt.replace(' ','')
        pattern = re.compile(r'“(.*?)”')
        result1 = pattern.findall(pt)

        if len(result1)==4:
            a,b,c,d = result1[2],result1[3],result1[0],result1[1]
            if textdict.get(a,-1)==-1:
                textdict[a] = [b]
            else:
                textdict[a].append(b)
        elif len(result1)==3:
            a,c,d = result1[2],result1[0],result1[1]
            if textdict.get(a,-1)==-1:
                textdict[a] = ['null']
        else:
            continue
    for pt in label:
        pt = pt.replace(' ','')
        pattern = re.compile(r'“(.*?)”')
        result1 = pattern.findall(pt)
        if len(result1)==4:
            a,b,c,d = result1[2],result1[3],result1[0],result1[1]
            if labeldict.get(a,-1)==-1:
                labeldict[a] = [b]
            else:
                labeldict[a].append(b)
        elif len(result1)==3:
            a,c,d = result1[2],result1[0],result1[1]
            if labeldict.get(a,-1)==-1:
                labeldict[a] = ['null']
        else:
            continue
    print(textdict)
    print(labeldict)
    for i in labeldict.keys():
        total+=1
        if textdict.get(i,-1)!=-1:
            right+=1
            for j in range(len(labeldict[i])):
                total+=1
                p = labeldict[i][j]
                if p in textdict[i]:
                    right+=1
    # print(total)
    # print(right)
    if total!=0:
        su = 1.0*right/total
    else:
        if right==0:
            su=1
        else:
            su=0
    print(su)
    return su
# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_args_into_dataclasses()[0]
model_args, data_args, training_args, finetuning_args = prepare_args()
model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="ppo")
# model = torch.nn.DataParallel(model)

data_collator = PPODataCollatorForChatGLM(
        tokenizer=tokenizer,
        min_input_length=data_args.max_source_length, # avoid truncating input sequences
        max_input_length=data_args.max_source_length,
        inference_mode=(not training_args.do_train)
    )
ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=max(training_args.per_device_train_batch_size // 4, 1),
        batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=int(training_args.num_train_epochs),
        max_grad_norm=training_args.max_grad_norm,
        remove_unused_columns=False
    )
def build_imdb_dataset(tokenizer, input_min_text_length=150, input_max_text_length=512):
    # load imdb with datasets
    # ds = load_dataset("imdb", split="train")
    # print(ds)
    def sp(sample):
        sample['prompt']= sample['text']
        sample['response']= sample['answer']
        return sample
    ds = load_dataset("json", data_files="data/dev_black.json", split="train")

    ds = ds.map(sp, batched=False)
    # ds.set_format(type="torch")
    return ds


data_collator = PPODataCollatorForChatGLM(
        tokenizer=tokenizer,
        min_input_length=data_args.max_source_length, # avoid truncating input sequences
        max_input_length=data_args.max_source_length,
        inference_mode=(not training_args.do_train)
    )

# set seed before initializing value head for deterministic eval
set_seed(43)


dataset = build_imdb_dataset(tokenizer)
dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="ppo")
# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)
# We retrieve the dataloader by calling the `build_dataset` function.

gen_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "logits_processor": get_logits_processor()
    }
generation_kwargs = {
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "logits_processor": get_logits_processor(),
                    "min_length": 5,
                    "max_new_tokens": 300,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 4,
                    "num_beams":6,
                    "return_dict_in_generate":True,
                    "output_scores":True
}
gen_kwargs = generation_kwargs
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)
ppo_trainer = PPOTrainerForChatGLM(
        training_args=training_args,
        finetuning_args=finetuning_args,
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
    )
# ppo_trainer = torch.nn.DataParallel(ppo_trainer)
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
output_length_sampler = LengthSampler(40, 80)
# sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device) 这个是reward

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.

for k in range(1):
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        reward=[]
        # batch只有input_ids
        # print(batch)
        # print(label)
        query_tensors = batch["input_ids"]
        label = batch['label']
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        # model.eval() 

        # responses_with_queries = ppo_trainer.generate(query_tensors,length_sampler=output_length_sampler,**gen_kwargs)
        generated_outputs = ppo_trainer.generate(query_tensors,max_length=450,**generation_kwargs)
        
        # print(generated_outputs)
        a = generated_outputs.sequences_scores
        generated_outputs = generated_outputs.sequences
        responses = generated_outputs[0][query_tensors.shape[1]:]
        # print(responses.shape)
        r = tokenizer.decode(responses).replace("<eop>", "")
        # print(r)
        genscore = compute_genscore(r,label)
        # responses = responses.unsqueeze(0)
        # responses = torch.tensor(responses,dtype=torch.long)
        if genscore<1.0:
            genscore=0
        a = a.cpu()
        a = list(a)
        print(a)
        ma = max(a)
        mi = min(a)
        b=[]
        for i in a:
            x = (i-mi)/(ma-mi+0.0000000001)
            b.append(x)
        a = b
        print(a)
        arr_var = np.var(a)
        sc = float(arr_var)+genscore
        # sc = float(arr_var)
        sc = torch.tensor(sc,dtype=torch.float64).to(device)
        reward.append(sc)
        reward = compute_rewards(reward,model,tokenizer)
        # Run PPO step
        print('sum:',sum(reward))
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        split_into_list = lambda x: [x[i] for i in range(len(x))]

        if epoch%40==0 and epoch==0:
            evaluate(model,ppo_trainer,tokenizer,gen_kwargs,dataset)
            break
        query_tensors = query_tensors.squeeze()
        responses_with_queries = query_tensors.squeeze()
        stats = ppo_trainer.step([query_tensors], [responses_with_queries], [reward])
        ppo_trainer.log_stats(stats, batch, reward)
        ppo_trainer.update_stats(stats, batch, reward)
        
        # evaluate(ppo_trainer,tokenizer)
        # stats = ppo_trainer.step(query_tensors, response_tensors, reward)
        # ppo_trainer.log_stats(stats, batch, reward)
        ppo_trainer.save_state() # along with the loss values
        ppo_trainer.save_model()
        
        # reinforce_eval.run()