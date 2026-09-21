#!/usr/bin/env python3
"""
从 Ascend/agent-skills 总仓抽取全部 skill，按【模块目录分类】打包离线仓库 skills.tar.gz。
name 保持原名，同名 skill 覆盖（后写覆盖先写）。

用法:
  python3 build_store.py <总仓路径> <输出tar路径>

输出结构（skills.tar.gz 解压后）:
  <模块>/<skill原名>/SKILL.md
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

def main():
    if len(sys.argv) != 3:
        print('用法: python3 build_store.py <总仓路径> <输出tar路径>', file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]
    out_tar = sys.argv[2]

    skills = collect_skills(src)
    entries = []
    for s in skills:
        mod = get_module(s)
        if not mod:
            print('跳过未映射模块:', s, file=sys.stderr)
            continue
        name = s.split('/')[-1]  # 原名
        entries.append({'src': s, 'module': mod, 'name': name})

    # 打 tar 到临时目录（覆盖策略）
    tmp = tempfile.mkdtemp(prefix='skillstore_')
    for e in entries:
        dst_dir = os.path.join(tmp, e['module'], e['name'])
        src_dir = os.path.join(src, e['src'].replace('/', os.sep))
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)  # 同名覆盖
        shutil.copytree(src_dir, dst_dir)

    with open(os.path.join(tmp, 'entries.json'), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)

    os.makedirs(os.path.dirname(out_tar), exist_ok=True)
    with tarfile.open(out_tar, 'w:gz') as tar:
        for name in sorted(os.listdir(tmp)):
            tar.add(os.path.join(tmp, name), arcname=name)

    shutil.rmtree(tmp)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(tmp) for f in fs) if os.path.exists(tmp) else 0
    print('打包完成: %d 个 skill' % len(entries))
    print('  tar: %s (%.2f MB)' % (out_tar, os.path.getsize(out_tar) / 1024 / 1024))

if __name__ == '__main__':
    main()