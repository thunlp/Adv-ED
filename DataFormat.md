# Data Format Specification
Due to the licence issues, we cannot share the source ACE2005 dataset or the preprocessed data.

So we specify the data format as follows and you can preprocess the data follow the format.

The data used in our model should be preprocessed to follow files (`Tag` means dataset tag, such as `ACE_Train`, `ACE_Test`, `DS_Train`):
## Tag_wordEmb.npy

### specification

Word embedding indices for input sentences, each sentence's length must be equal to `SenLen` in `constant.py`. The padding index is `0`.

Its shape should be `(BatchSize, SenLen)`.

### example
`[[1,2,3,4,5,6],[2,4,2,1,4,0]]`

## Tag_posEmb.npy

### specification

Relative position embedding indices to the candidate trigger for each instance. For instance, for a token in the tokenized sentence, its position embedding
 index is `TokenPosition-TriggerPosition+SenLen`. The position embedding index for padding token is `0`.
 
Its shape should be `(BatchSize, SenLen)`.

### example

`[[4,5,6,7,8,9],[2,3,4,5,6,0]]`

## Tag_local.npy (DMCNN only)

### specification

Lexical feature for DMCNN. It is the word embedding indices for the words whose distance to the candidate trigger is less than `LocalLen` in `constant.py`.

Its shape should be `(BatchSize, 2*LocalLen+1)`.
### example

`[[2,3,4],[1,4,0]]`

## Tag_label.npy

### specification

Event type label for each instance, the `NA` is `0`.

Its shape should be `(BatchSize)`.

### example

`[1,0]`

## Tag_maskL.npy

### specification

Mask matrix for the words on the left of the candidate trigger.

Its shape should be `(BatchSize,SenLen)`.

### example

`[[1.,1.,1.,0.,0.,0.],[1.,1.,1.,1.,1.,0.]]`

## Tag_maskR.npy

### specification

Mask matrix for the words on the right of the candidate trigger except the padding tokens.

Its shape should be `(BatchSize,SenLen)`.

### example

`[[0.,0.,0.,1.,1.,1.],[0.,0.,0.,0.,0.,0.]]`

## Tag_inMask.npy (DMBERT only)

### specification

The input mask for `BERT`, which includes all words except the padding tokens.

Its shape should be `(BatchSize,SenLen)`.

### example

`[[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,0.]]`

## Tag_index.npy (optional)

### specification

The overall index for each instance, which is used to find corresponding original data item. (after random shuffle)

If you want it, uncomment corresponding lines in `dataset.py`.

Its shape should be `(BatchSize)`.

### example

`[1,2]`
