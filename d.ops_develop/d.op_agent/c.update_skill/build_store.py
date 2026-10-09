#!/usr/bin/env python3
"""
从 Ascend/agent-skills 总仓抽取全部 skill，按【模块目录分类】打包离线仓库 skills.tar.gz。

关键规则（与 opencode 的 skill 规范对齐）：
  - skill 的最终目录名取 SKILL.md 的 `name` 字段（不是上游目录名），
    保证「目录名 == name」；同名按后写覆盖先写。
  - name 需满足 `^[a-z0-9]+(-[a-z0-9]+)*$`，否则做规整（小写、非法字符转连字符、
    取最后一个 `/` 段、合并多余连字符），规整后仍非法则跳过。
  - 打包时同步把 SKILL.md 的 name 行改写成最终目录名，保证一致。

用法:
  python3 build_store.py <总仓路径> <输出tar路径>

输出结构（skills.tar.gz 解压后）:
  <模块>/<skillname>/SKILL.md
  entries.json
"""
import os, re, json, shutil, sys, tarfile, tempfile

MODULE_MAP = {
    ('official', 'CANNBot'): 'cannbot',
    ('official', 'PyTorch'): 'pytorch',
    ('official', 'Common'): 'common',
    ('official', 'MindStudio'): 'mindstudio',
    ('official', 'MindSpeed'): 'mindspeed',
    ('official', 'MindCluster'): 'mindcluster',
    ('official', 'verl'): 'verl',
    ('official', 'MindSeriesSDK'): 'mindseries-sdk',
    ('official', 'vllm-ascend'): 'vllm-ascend',
    ('community', 'Op'): 'community-op',
    ('community', 'Tools'): 'community-tools',
}

NAME_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')

def get_module(rel):
    parts = rel.split('/')
    key = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], '')
    return MODULE_MAP.get(key)

def collect_skills(src):
    skills = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if 'SKILL.md' in files:
            rel = os.path.relpath(root, src).replace(os.sep, '/')
            skills.append(rel)
    return sorted(skills)

def read_name(skill_dir):
    """读取 SKILL.md 的 name 字段（兼容 BOM）；没有则返回 None。"""
    sk = os.path.join(skill_dir, 'SKILL.md')
    try:
        txt = open(sk, encoding='utf-8-sig', errors='replace').read()
    except OSError:
        return None
    m = re.search(r'---\s*\n(.*?)\n---', txt, re.S)
    if not m:
        return None
    nm = re.search(r'^name:\s*(.+?)\s*$', m.group(1), re.M)
    if not nm:
        return None
    return nm.group(1).strip().strip('"\'')

def canonical_name(raw_name, fallback):
    """规整出合法的 opencode skill 名；非法则返回 None。"""
    n = (raw_name or '').strip()
    if not n:
        n = fallback
    n = n.split('/')[-1].lower()
    n = re.sub(r'[^a-z0-9]+', '-', n)
    n = re.sub(r'-+', '-', n).strip('-')
    return n if NAME_RE.fullmatch(n) else None

def rewrite_sk_name(skill_dir, name):
    sk = os.path.join(skill_dir, 'SKILL.md')
    txt = open(sk, encoding='utf-8-sig', errors='replace').read()
    txt = re.sub(r'^(name:\s*).*?(\s*)$', r'\g<1>' + name + r'\g<2>',
                 txt, count=1, flags=re.M)
    open(sk, 'w', encoding='utf-8').write(txt)

def main():
    if len(sys.argv) != 3:
        print('用法: python3 build_store.py <总仓路径> <输出tar路径>', file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]
    out_tar = sys.argv[2]

    skills = collect_skills(src)
    entries = []
    used = set()  # (module, name) 去重
    for s in skills:
        mod = get_module(s)
        if not mod:
            print('跳过未映射模块:', s, file=sys.stderr)
            continue
        raw = read_name(os.path.join(src, s.replace('/', os.sep)))
        name = canonical_name(raw, s.split('/')[-1])
        if not name:
            print('跳过（name 非法）:', mod, s, file=sys.stderr)
            continue
        if (mod, name) in used:
            print('跳过（同名）:', mod, name, '<-', s, file=sys.stderr)
            continue
        used.add((mod, name))
        entries.append({'src': s, 'module': mod, 'name': name, 'raw_name': raw})

    tmp = tempfile.mkdtemp(prefix='skillstore_')
    for e in entries:
        src_dir = os.path.join(src, e['src'].replace('/', os.sep))
        dst_dir = os.path.join(tmp, e['module'], e['name'])
        shutil.copytree(src_dir, dst_dir)
        rewrite_sk_name(dst_dir, e['name'])

    with open(os.path.join(tmp, 'entries.json'), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)

    os.makedirs(os.path.dirname(out_tar), exist_ok=True)
    with tarfile.open(out_tar, 'w:gz') as tar:
        for name in sorted(os.listdir(tmp)):
            tar.add(os.path.join(tmp, name), arcname=name)

    shutil.rmtree(tmp)
    print('打包完成: %d 个 skill' % len(entries))
    print('  tar: %s (%.2f MB)' % (out_tar, os.path.getsize(out_tar) / 1024 / 1024))

if __name__ == '__main__':
    main()
