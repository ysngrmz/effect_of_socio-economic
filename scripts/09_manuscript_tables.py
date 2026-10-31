"""Construct human-readable table drafts; no automatic claims of significance."""
import pandas as pd
from common import OUT
with open(OUT/'manuscript_table_drafts.md','w',encoding='utf-8') as f:
 f.write('# Draft Results Tables — verify before submission\n\n')
 for file,title in [('gpa_summary.csv','GPA descriptive summary'),('main_summary.csv','Nested-CV performance'),('ablation_summary.csv','Feature-group ablation'),('paired_descriptive_comparisons.csv','Paired descriptive comparisons')]:
  path=OUT/file
  if path.exists():
   frame=pd.read_csv(path)
   f.write(f'## {title}\n\n');f.write(frame.to_markdown(index=False));f.write('\n\n')
 f.write('## Required manual decisions\n\n- Confirm GPA outlier handling and first-year GPA provenance.\n- Confirm BALLI subscale mapping and reverse coding.\n- Confirm original 100 vs 200 repeats and original splits.\n- Describe performance differences descriptively, not as statistical significance without valid inference.\n- Do not equate overall GPA with English language proficiency.\n- Do not describe sensitivity/permutation rankings as causality.\n')
print('Wrote manuscript_table_drafts.md')
