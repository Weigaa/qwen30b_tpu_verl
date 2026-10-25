#!/usr/bin/env python3
"""Submit a text prompt using the same local tokenizer, without loading weights."""
import argparse
import json
from urllib.request import Request,urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8099')
    parser.add_argument('--tokenizer',default='/data/Qwen3-30B-A3B')
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--rank',type=int)
    parser.add_argument('--policy-version',default='current')
    parser.add_argument('--auth-token')
    args=parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=True)
    tokens=tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
                                        tokenize=True,add_generation_prompt=True)
    job=dict(prompt_token_ids=tokens,max_tokens=args.max_tokens,policy_version=args.policy_version)
    if args.rank is not None:job['rank']=args.rank
    headers={'Content-Type':'application/json'}
    if args.auth_token:headers['Authorization']='Bearer '+args.auth_token
    req=Request(args.url+'/v1/tailserve/requests',data=json.dumps(job).encode(),headers=headers)
    with urlopen(req,timeout=30) as response:result=json.load(response)
    print(json.dumps(result,indent=2))
    print('Poll:',args.url+'/v1/tailserve/requests/'+result['id'])


if __name__=='__main__':main()
