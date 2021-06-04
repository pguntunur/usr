import random
import json
import csv
import numpy as np
import os
import regression
import spacy

SRC_FCT_FOLDER_PATH = "ap_data/chateval_src_fct/"

# Eval prep
nlp = spacy.load('en')
def tokenize(data):
  new_data = []
  print("Tokenizing")
  data = [s.replace("_go", "").replace("_eos", "").strip() for s in data]
  docs = nlp.tokenizer.pipe([' '.join(s.lower().split()) for s in data])
  for doc in docs:
    # Tokenize with spacy
    tokenized = ' '.join([e.text for e in doc])

    # Fix mis-tokenized tags
    tokenized = "_go " + tokenized + " _eos"
    new_data.append(tokenized)

  return new_data

def prep_mlm(fn, model_num):
  outputs = tokenize(open(SRC_FCT_FOLDER_PATH + fn).readlines())
  valid_src = [e.strip().split("_eos ")[-1] for e in open(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".src").readlines()]
  output_lines = [s + " " + r + "\n" for s,r in zip(valid_src, outputs)]
  open("undr/" + fn, "w+").writelines([' '.join(e.split()) + "\n" for e in output_lines])


def get_scores(fn, model_num):
  fn = "valid_freq" + model_num + ".fct"
  prep_mlm(fn, model_num)

  scores = {}
  fn_base = fn.split(".")[0]
  
  # MLM
  mlm = """
export EVAL_FILE=undr/{0}.txt

CUDA_VISIBLE_DEVICES=1 python3 run_lm_finetuning.py \
    --per_gpu_eval_batch_size=1 \
    --output_dir=roberta_ft \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --train_data_file=$EVAL_FILE \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --mlm \
    --overwrite_cache 
""".format(fn_base)
  print(mlm)
  os.system(mlm)

  mlm_scores = eval(open("undr/{0}.scores".format(fn_base)).read())
  scores["USR-MLM"] = np.mean(mlm_scores)

  
  print(len(mlm_scores))
  # Regression
#   regr_scores = regression.scores(mlm_scores, drc_scores, drf_scores)
#   scores['USR'] = np.mean(regr_scores)

#   print(scores)
  with open("ap_data/outputs/output_" + model_num + ".txt", 'w') as convert_file:
     convert_file.write(json.dumps(mlm_scores))
  
  return scores


model_num = input("Enter model number (0 for all): ")
if (os.path.exists(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".src") and os.path.exists(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".fct")):
  print("Model exists, files found.")
  get_scores("Transformer-baseline_v0.txt", model_num)
elif (int(model_num) == 0):
  possible_model_filenames = [f for f in os.listdir(SRC_FCT_FOLDER_PATH) if os.path.isfile(os.path.join(SRC_FCT_FOLDER_PATH, f))]
  
  # we can do what we do below because all the model files are named consistently.
  possible_model_nums = [ele[10:] for ele in possible_model_filenames]
  possible_model_nums = [ele[:-4] for ele in possible_model_nums]
  possible_model_nums = [ele for ele in possible_model_nums if ele.isdigit()]
  possible_model_nums = [int(ele) for ele in possible_model_nums]
  possible_model_nums.sort()
  possible_model_nums = list(set(possible_model_nums))

  for mn in possible_model_nums:
    get_scores("Transformer-baseline_v0.txt", str(mn))
    print("Model #" + str(mn) + " complete.")

else:
  print("Model number does not exist")
