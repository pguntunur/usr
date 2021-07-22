
# Normalization function for Pandas from Prof's NB
def normalize_df(dataset_name, df, ds_meta):
    dataset_meta = ds_meta[dataset_name]
    for annotation in dataset_meta['annotations']:
        df['annotations.' + annotation] = df['annotations.' + annotation].apply(dataset_meta['aggregation'])
    df.context = df.context.str.replace('U: ','').replace('A: ','').replace('B: ', '').replace('Speaker 1: ','').replace('Speaker 2: ','')
    df.response = df.response.str.replace('S: ','').replace('A: ','').replace('B: ', '').replace('Speaker 1: ','').replace('Speaker 2: ','')
    df.reference = df.reference.apply(lambda l: [x.replace('S: ','').replace('A: ','').replace('B: ', '').replace('Speaker 1: ','').replace('Speaker 2: ','') for x in l])
    return df

# DailyDialog - Gupta
fn = "dailydialog-gupta_eval.json"
dataset = "dailydialog-gupta"

with open(fn) as f:
    df = pd.json_normalize(json.load(f))
# NOTE: removing lines with no reference
df = df[df.reference.apply(len)>0]

df = normalize_df(dataset, df, dataset_meta_info)
