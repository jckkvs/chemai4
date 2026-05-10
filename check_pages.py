import sys
sys.path.insert(0, 'c:/Users/horie/chemai2')

# 新しいパス構成でチェック
pages = [
    ('automl_page',       'frontend_streamlit.pages.automl_page'),
    # pipeline サブパッケージ
    ('data_load_page',    'frontend_streamlit.pages.pipeline.data_load_page'),
    ('eda_page',          'frontend_streamlit.pages.pipeline.eda_page'),
    ('preprocess_page',   'frontend_streamlit.pages.pipeline.preprocess_page'),
    ('evaluation_page',   'frontend_streamlit.pages.pipeline.evaluation_page'),
    ('interpret_page',    'frontend_streamlit.pages.pipeline.interpret_page'),
    ('dim_reduction_page','frontend_streamlit.pages.pipeline.dim_reduction_page'),
    # tools サブパッケージ
    ('chem_page',         'frontend_streamlit.pages.tools.chem_page'),
]

ok = 0
fail = 0
for name, mod in pages:
    try:
        __import__(mod)
        print('OK: ' + name)
        ok += 1
    except Exception as e:
        print('FAIL: ' + name + ' -- ' + str(e))
        fail += 1

print('')
print(str(ok) + '/' + str(ok + fail) + ' pages import OK')
print('')
print('Structure:')
print('  pages/')
print('    automl_page.py        (Main - full pipeline)')
print('    pipeline/             (AutoML sub-steps)')
print('      data_load_page.py')
print('      eda_page.py')
print('      preprocess_page.py')
print('      evaluation_page.py')
print('      interpret_page.py')
print('      dim_reduction_page.py')
print('    tools/                (Standalone tools)')
print('      chem_page.py')
