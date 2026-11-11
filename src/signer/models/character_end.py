"""Independent character-end adapter inspired by character/subword modeling.

Not the original CharBERT implementation. The frozen NER predicts type/start;
this adapter can only choose a nonempty end inside its final covering token.
No gold, entity dictionary or identifier is accepted by the inference module.
"""
from collections import Counter
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence

LABELS=('PER','LOC','ORG','MISC')


def make_vocabulary(texts):
    return {'<PAD>':0,'<UNK>':1,'<BOS>':2,'<EOS>':3,
            **{char:index+4 for index,char in enumerate(sorted(set(''.join(texts))))}}


def endpoint_feature(text,language,label,span_start,token_offset,vocabulary,language_ids,context=16):
    left,right=map(int,token_offset)
    if not 0<=left<right<=len(text) or not 0<=span_start<right:
        raise ValueError('Invalid predicted character coordinates')
    first=max(left+1,span_start+1)
    positions=list(range(first,right+1))
    if not positions:raise ValueError('No nonempty end candidate')
    begin=max(0,left-context);end=min(len(text),right+context)
    return {'char_ids':[2]+[vocabulary.get(c,1) for c in text[begin:end]]+[3],
        'boundary_indices':[p-begin for p in positions],
        'positions':positions,'keep_index':len(positions)-1,
        'language_id':language_ids.get(language,0),'label_id':LABELS.index(label)}


def training_endpoints(row,owner_function,vocabulary,language_ids):
    """Gold is used only to construct TRAIN targets, not inference features.

    Retain entities with previously unowned BIO tokens if their end is covered.
    Uncovered/truncated ends are explicitly counted, never removed from DEV gold.
    """
    from types import SimpleNamespace
    entities=[SimpleNamespace(start=a,end=b,label=c) for a,b,c in row['raw']]
    offsets=row['offsets'];result=[];audit=Counter(raw=len(entities))
    for index,entity in enumerate(entities):
        owned=[i for i,(a,b) in enumerate(offsets) if owner_function(a,b,entities)==index]
        token=max(owned,key=lambda i:(offsets[i][1],i)) if owned else None
        if token is None or not offsets[token][0]<entity.end<=offsets[token][1]:
            candidates=[i for i,(a,b) in enumerate(offsets) if a<entity.end<=b]
            if not candidates:audit['uncovered_end']+=1;continue
            token=min(candidates,key=lambda i:(offsets[i][1]-offsets[i][0],-i))
            audit['fallback_covering_token']+=1
        feature=endpoint_feature(row['text'],row['language'],entity.label,entity.start,
                                  offsets[token],vocabulary,language_ids)
        target=feature['positions'].index(entity.end)
        result.append({'feature':feature,'token_index':token,'target':target})
        audit['covered']+=1;audit['intra_token_end']+=int(entity.end<offsets[token][1])
    return result,dict(audit)


def prediction_endpoints(text,language,offsets,tag_ids,decode,vocabulary,language_ids):
    """Uses predicted BIO and text offsets only; signature has no gold input."""
    if len(offsets)!=len(tag_ids):raise ValueError('Token prediction identity mismatch')
    valid=[i for i,(a,b) in enumerate(offsets) if b>a]
    compact=decode([tag_ids[i] for i in valid]);unique={}
    for first,last,label in compact:
        indices=valid[first:last]
        if not indices:raise ValueError('Empty predicted BIO interval')
        left=min(offsets[i][0] for i in indices);right=max(offsets[i][1] for i in indices)
        token=max(indices,key=lambda i:(offsets[i][1],i))
        key=(left,right,label)
        if key not in unique or token>unique[key]:unique[key]=token
    result=[]
    for (left,right,label),token in sorted(unique.items()):
        result.append({'feature':endpoint_feature(text,language,label,left,offsets[token],vocabulary,language_ids),
                       'token_index':token,'base_span':[left,right,label]})
    return result


def pack_features(features,token_states,device):
    if not features or len(features)!=len(token_states):raise ValueError('Empty/mismatched feature batch')
    lengths=torch.tensor([len(f['char_ids']) for f in features],dtype=torch.long)
    chars=pad_sequence([torch.tensor(f['char_ids'],dtype=torch.long) for f in features],batch_first=True,padding_value=0)
    size=max(len(f['boundary_indices']) for f in features)
    indices=torch.zeros(len(features),size,dtype=torch.long)
    mask=torch.zeros(len(features),size,dtype=torch.bool);keep=mask.clone()
    for i,feature in enumerate(features):
        n=len(feature['boundary_indices']);indices[i,:n]=torch.tensor(feature['boundary_indices'])
        mask[i,:n]=True;keep[i,feature['keep_index']]=True
    return {'char_ids':chars.to(device),'lengths':lengths,'candidate_indices':indices.to(device),
        'candidate_mask':mask.to(device),'keep_mask':keep.to(device),
        'language_ids':torch.tensor([f['language_id'] for f in features],device=device),
        'label_ids':torch.tensor([f['label_id'] for f in features],device=device),
        'token_states':token_states.detach().to(device)}


class CharacterEndHead(nn.Module):
    def __init__(self,vocabulary_size,language_count,hidden_size=768):
        super().__init__()
        self.characters=nn.Embedding(vocabulary_size,64,padding_idx=0)
        self.encoder=nn.GRU(64,64,batch_first=True,bidirectional=True)
        self.text_projection=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,64),nn.GELU())
        self.languages=nn.Embedding(language_count,8);self.types=nn.Embedding(4,8)
        self.scorer=nn.Sequential(nn.Linear(336,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,1))
        nn.init.zeros_(self.scorer[-1].weight);nn.init.zeros_(self.scorer[-1].bias)
        self.keep_bias=nn.Parameter(torch.tensor(2.))

    def forward(self,char_ids,lengths,candidate_indices,candidate_mask,keep_mask,language_ids,label_ids,token_states):
        if not bool(candidate_mask.any(1).all()):raise ValueError('No valid character end')
        packed=pack_padded_sequence(self.characters(char_ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed)
        encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=char_ids.size(1))
        gather=candidate_indices.unsqueeze(-1).expand(-1,-1,128)
        before=encoded.gather(1,gather);after=encoded.gather(1,gather+1)
        fixed=torch.cat([self.text_projection(token_states),self.languages(language_ids),self.types(label_ids)],-1)
        features=torch.cat([before,after,fixed[:,None,:].expand(-1,before.size(1),-1)],-1)
        logits=self.scorer(features).squeeze(-1)+self.keep_bias*keep_mask
        return logits.masked_fill(~candidate_mask,torch.finfo(logits.dtype).min)


def refine_spans(head,specs,token_states,device):
    if not specs:return []
    features=[r['feature'] for r in specs]
    selected=head(**pack_features(features,token_states,device)).argmax(-1).cpu().tolist()
    result=set()
    for spec,index in zip(specs,selected):
        left,_,label=spec['base_span'];end=spec['feature']['positions'][index]
        if end<=left:raise ValueError('Refiner produced an empty span')
        result.add((left,end,label))
    return [list(s) for s in sorted(result)]
