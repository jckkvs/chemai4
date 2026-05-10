"""mutatest API の mutant_code 属性を正確に調べる"""
from mutatest.api import Genome
from mutatest.transformers import CATEGORIES
from mutatest.filters import CategoryCodeFilter
from pathlib import Path

g = Genome(source_file=Path('domainml/constraints/engine.py'))
targets = list(g.targets)

for t in targets[:4]:
    op_code = CATEGORIES.get(t.ast_class)
    if not op_code:
        continue
    valid_muts = list(CategoryCodeFilter(codes=(op_code,)).valid_mutations)
    if not valid_muts:
        continue
    mut = g.mutate(t, valid_muts[0], write_cache=False)
    mc = getattr(mut, 'mutant_code', None)
    has_body = hasattr(mc, 'body')
    print(f"{t.ast_class} L{t.lineno}: type={type(mc).__name__} has_body={has_body}")
    if isinstance(mc, bytes):
        print(f"  -> bytes len={len(mc)}")
    elif mc is not None and not has_body:
        attrs = [a for a in dir(mc) if not a.startswith('_')][:8]
        print(f"  -> attrs: {attrs}")
