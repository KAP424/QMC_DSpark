import json

nb = json.load(open('/home/dev/KAP/test/qwen3.5-0.8b-base_quickstart.ipynb'))

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

def src(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

MD = '''## 5. 实测：Countdown / 1D-ARC —— 模型回答 vs 正确答案、正确率

直接在代码里从两个数据集**随机抽样**（固定种子，不落盘中间文件）。判定全部程序化：

- **Countdown**（`data/countdown/data/train-*.parquet`，`{target, nums}`）：从回答里抽取算式，要求**恰好用完给定数字**（多重集合相同），求值与 target 比对；正确答案由程序暴力搜索。
- **1D-ARC**（`data/1d_arc/1D-ARC-main/dataset/*/*.json`，演示对 + test 查询）：解析与网格等宽的整数行，与 ground-truth 网格**精确匹配**（另附逐格准确率供参考，注意抄输入就能拿高分）。

5.1 先看 2+2 题的完整回答对照，5.2 每个数据集抽 20 题测正确率。'''

CODE1 = '''import glob, json, os, random, re
from fractions import Fraction

COUNTDOWN = '/home/dev/KAP/data/countdown/data/train-00000-of-00001.parquet'
ARC_DIR = '/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset'

def sample_countdown(n, seed):
    import pyarrow.parquet as pq
    t = pq.read_table(COUNTDOWN)
    rng = random.Random(seed)
    out = []
    for i in rng.sample(range(t.num_rows), n):
        tgt = t.column('target')[i].as_py()
        nums = t.column('nums')[i].as_py()
        prompt = (f'Using the numbers {nums}, create an expression that equals {tgt}. '
                  'You can use basic arithmetic operations (+, -, *, /). '
                  'Each number must be used exactly once.')
        out.append({'dataset': 'countdown', 'id': f'row{i}', 'prompt': prompt,
                    'raw': {'nums': nums, 'target': tgt}})
    return out

def sample_arc(n, seed):
    files = sorted(glob.glob(f'{ARC_DIR}/*/*.json'))
    rng = random.Random(seed)
    out = []
    for f in rng.sample(files, n):
        task = json.load(open(f))
        demos, test = task['train'], task['test'][0]
        row = lambda g: ' '.join(map(str, g[0]))
        parts = ['Apply the same transformation rule to the test input.',
                 'Here are example input/output pairs:']
        for dmo in demos:
            parts += ['Input:', row(dmo['input']), 'Output:', row(dmo['output'])]
        parts += ['Now, give the output for the test input.', 'Test Input:', row(test['input']), 'Output:']
        out.append({'dataset': '1d_arc', 'id': '/'.join(f.split('/')[-2:]),
                    'prompt': chr(10).join(parts), 'raw': {'test': test}})
    return out

def countdown_truth(nums, target):
    """暴力搜索一个恰好用完 nums 且等于 target 的算式，返回表达式字符串或 None。"""
    def gen(items):
        if len(items) == 1:
            yield items[0]
            return
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x, sx = items[i]
                y, sy = items[j]
                others = [items[k] for k in range(n) if k != i and k != j]
                for v, s in ((x + y, f'({sx}+{sy})'), (x - y, f'({sx}-{sy})'),
                             (x * y, f'({sx}*{sy})')) + (((x / y, f'({sx}/{sy})'),) if y != 0 else ()):
                    yield from gen(others + [(v, s)])
    for v, s in gen([(Fraction(n_), str(n_)) for n_ in nums]):
        if v == target:
            return s
    return None

def check_countdown(resp, nums, target):
    """-> (correct: bool, detail: str)。正确 = 存在恰好用齐数字且值为 target 的算式。"""
    cands = re.findall(r'[\\d\\(\\)\\+\\-\\*\\/][\\d\\(\\)\\+\\-\\*\\/\\s]*[\\d\\)]', resp)
    tried = []
    for c in cands:
        if not re.search(r'[\\+\\-\\*\\/]', c):
            continue
        runs = re.findall(r'\\d+', c)
        if sorted(map(int, runs)) != sorted(nums):
            continue
        try:
            val = eval(c, {'__builtins__': {}}, {})
        except Exception:
            continue
        if isinstance(val, float) and abs(val - round(val)) < 1e-9:
            val = round(val)
        if isinstance(val, int):
            tried.append((re.sub(r'\\s+', '', c), val))
    for expr, val in tried:
        if val == target:
            return True, f'{expr} = {val}'
    if tried:
        return False, '用齐数字但结果错: ' + '; '.join(f'{e}={v}' for e, v in tried[:2]) + f' (target {target})'
    return False, '未找到恰好用齐数字的算式'

def check_arc(resp, test_out):
    """-> (correct: bool, detail: str)。正确 = 解析到等宽行且与 truth 精确一致。"""
    truth = test_out[0]
    width = len(truth)
    for line in resp.splitlines():
        xs = re.findall(r'\\d+', line)
        if len(xs) == width:
            cand = list(map(int, xs))
            if cand == truth:
                return True, '精确匹配'
            acc = sum(a == b for a, b in zip(cand, truth)) / width
            return False, f'不匹配（逐格 {acc:.0%}，抄输入也会高，须精确匹配）'
    return False, f'未解析到宽度 {width} 的整数行'

print('samplers ready')'''

CODE2 = '''def gen_response(prompt, max_new_tokens=512):
    ids = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')],
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

# ---- 5.1 四题完整对照 ----
for it in sample_countdown(2, 0) + sample_arc(2, 0):
    resp = gen_response(it['prompt'])
    print('=' * 76)
    print(f"[{it['dataset']}] {it['id']}")
    print('PROMPT  :', it['prompt'][:150].replace('\\n', ' | '), '...' if len(it['prompt']) > 150 else '')
    print('RESPONSE:', resp[:400].replace('\\n', ' | '), '...' if len(resp) > 400 else '')
    if it['dataset'] == 'countdown':
        nums, target = it['raw']['nums'], it['raw']['target']
        print('正确示例:', countdown_truth(nums, target), '=', target)
        ok, detail = check_countdown(resp, nums, target)
        print('判定    :', ('正确 ' if ok else '错误 ') + detail)
    else:
        ok, detail = check_arc(resp, it['raw']['test']['output'])
        print('正确输出:', ' '.join(map(str, it['raw']['test']['output'][0])))
        print('判定    :', ('正确 ' if ok else '错误 ') + detail)'''

CODE3 = '''# ---- 5.2 正确率：每个数据集 20 题（种子不同，与 5.1 不重叠不保证）----
N = 20
summary = {}
for name, sampler in (('countdown', sample_countdown), ('1d_arc', sample_arc)):
    items = sampler(N, 123)
    ok_n, lines = 0, []
    for k, it in enumerate(items):
        resp = gen_response(it['prompt'])
        if it['dataset'] == 'countdown':
            ok, detail = check_countdown(resp, it['raw']['nums'], it['raw']['target'])
        else:
            ok, detail = check_arc(resp, it['raw']['test']['output'])
        ok_n += ok
        lines.append(f"  {'V' if ok else 'X'} {it['id'][:28]:28s} {detail[:70]}")
        if (k + 1) % 10 == 0:
            print(f'  ... {name} {k + 1}/{N}', flush=True)
    summary[name] = ok_n
    print(f'\\n[{name}] 正确率 {ok_n}/{N} = {ok_n / N:.0%}')
    print('\\n'.join(lines))
    print()
print('汇总:', {k: f'{v}/{N} = {v / N:.0%}' for k, v in summary.items()})'''

cells = nb['cells']
i_md = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and '实测：Countdown' in src(c))
i_c1 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'SEL = json.load' in src(c))
i_c2 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'def gen_response' in src(c))
assert i_c1 == i_md + 1 and i_c2 == i_md + 2, (i_md, i_c1, i_c2)
cells[i_md] = md(MD)
cells[i_c1] = code(CODE1)
cells[i_c2] = code(CODE2)
cells.insert(i_c2 + 1, code(CODE3))
json.dump(nb, open('/home/dev/KAP/test/qwen3.5-0.8b-base_quickstart.ipynb', 'w'), indent=1, ensure_ascii=False)
print('section 5 rebuilt: md + sampler/checker + demo + accuracy cells')
