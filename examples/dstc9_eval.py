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
  outputs = tokenize(open(fn).readlines())
  valid_src = [e.strip().split("_eos ")[-1] for e in open(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".src").readlines()]
  output_lines = [s + " " + r + "\n" for s,r in zip(valid_src, outputs)]
  open("undr/" + fn, "w+").writelines([' '.join(e.split()) + "\n" for e in output_lines])

def prep_both(fn, model_num):
  outputs = tokenize(open(fn).readlines())
  valid_src = [e.strip().split("_eos ")[-1] for e in open(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".src").readlines()]
  valid_fct = [e.strip() for e in open(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".fct").readlines()] 

  valid_ctx = [s+" " +f+" _eos" for  s,f in zip(valid_src, valid_fct)]

  rows = [(0,1,2,c,o,0) for c,o in zip(valid_ctx, outputs)]
  rows = [rows[0]] + rows
  folder = "both/" + fn.split(".")[0] + "/"
  if not os.path.exists(folder):
    os.mkdir(folder)
  csv.writer(open(folder + "/dev.tsv", "w+"),  delimiter="\t").writerows(rows)
  csv.writer(open(folder + "/train.tsv", "w+"), delimiter="\t").writerows(rows)

def prep_uk(fn, model_num):
  outputs = tokenize(open(fn).readlines())

  valid_fct = [e.strip() for e in open(SRC_FCT_FOLDER_PATH + "valid_freq" + model_num + ".fct").readlines()] 

  valid_ctx = [f+" _eos" for f in valid_fct]

  rows = [(0,1,2,c,o,0) for c,o in zip(valid_ctx, outputs)]
  rows = [rows[0]] + rows
  folder = "fct/" + fn.split(".")[0] + "/"
  if not os.path.exists(folder):
    os.mkdir(folder)
  csv.writer(open(folder + "/dev.tsv", "w+"),  delimiter="\t").writerows(rows)
  csv.writer(open(folder + "/train.tsv", "w+"), delimiter="\t").writerows(rows)

def prep_bs(fn):
  outputs = tokenize(open(fn).readlines())
  #new = [e.replace("_go", "").replace("_eos", "").strip() for e in outputs]
  new = outputs
  open("bs_data/" + fn, "w+").writelines([e+"\n" for e in new])

def get_scores(fn, model_num):
  prep_mlm(fn, model_num)
  prep_bs(fn)
  prep_both(fn, model_num)
  prep_uk(fn, model_num)

  scores = {}
  fn_base = fn.split(".")[0]

  # BERTscore 
  # data_fn = "bs_data/" + fn
  # os.system("/usr/bin/python3 run_bertscore.py " + data_fn)
  # scores_fn = "bs_data/" + fn_base + ".scores"
  # scores['BERTScore'] = np.mean(eval(open(scores_fn).read()))

  #  METEOR
  # scores_fn = "bs_data/" + fn_base + ".meteor"
  # scores['METEOR'] = np.mean(eval(open(scores_fn).read()))

  # USR
  
  # DR:
  drc = """
export DATA_DIR=both/{0}/

CUDA_VISIBLE_DEVICES=1 python3 train_understandable.py \
    --per_gpu_eval_batch_size=1 \
    --output_dir=ctx \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --data_dir $DATA_DIR \
    --do_eval \
    --task_name=qqp
  """.format(fn_base)

  os.system(drc)
  drc_scores = eval(open("both/{0}/dr.scores".format(fn_base)).read())
  scores['USR-DRc'] = np.mean(drc_scores)

  drf = """
export DATA_DIR=fct/{0}/

CUDA_VISIBLE_DEVICES=1 python3 train_understandable.py \
    --per_gpu_eval_batch_size=1 \
    --output_dir=uk \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --data_dir $DATA_DIR \
    --do_eval \
    --task_name=qqp
""".format(fn_base)
  os.system(drf)
  drf_scores = eval(open("fct/{0}/dr.scores".format(fn_base)).read())
  scores['USR-DRf'] = np.mean(drf_scores)

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

  
  print(len(mlm_scores), len(drc_scores), len(drf_scores))
  # Regression
  regr_scores = regression.scores(mlm_scores, drc_scores, drf_scores)
  scores['USR'] = np.mean(regr_scores)

  print(scores)
  with open("ap_data/outputs/output_" + model_num + ".txt", 'w') as convert_file:
     convert_file.write(json.dumps(scores))
  
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
