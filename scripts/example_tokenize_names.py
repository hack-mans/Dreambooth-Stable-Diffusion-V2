!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-L-14')

model.tokenizer = model._first_module().processor.tokenizer
tokenizer = model._first_module().processor.tokenizer

id_to_vec = model._first_module().model.text_model.embeddings.token_embedding.weight
vocab_to_id = tokenizer.get_vocab()
id_to_vocab = {id:vocab for vocab, id in vocab_to_id.items()}

tokenizer.tokenize("danny devito") #three tokens 'danny</w>', 'dev', 'ito</w>'

#</w> tokens can be rendered seperately, no access to non </w> tokens currently expect as part of a larger word

#token ids
danny_id = vocab_to_id['danny</w>']
dev_id = vocab_to_id['dev']
ito_id = vocab_to_id['ito</w>']

cumberbatch_id = vocab_to_id['cumberbatch</w>']

#print the three ids
print(danny_id, dev_id, ito_id)

#vector for the tokens without position embedding information
token_vectors = id_to_vec[[danny_id, dev_id, ito_id]]
first_token_vector = id_to_vec[[danny_id]]

#averaging the three tokens, see if this renders the same
mean_vec = id_to_vec[[danny_id, dev_id, ito_id]].sum(axis=0)/3.

