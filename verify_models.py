import pickle
import os

models_dir = 'new/models'
models = sorted([f for f in os.listdir(models_dir) if f.endswith('.sav')])

print('='*70)
print(f'TOTAL MODELS: {len(models)}')
print('='*70)

for m in models:
    try:
        d = pickle.load(open(os.path.join(models_dir, m), 'rb'))
        acc = d.get('accuracy', 'N/A')
        cv = d.get('cv_accuracy', 'N/A')
        if isinstance(acc, float):
            acc = f'{acc*100:.1f}%'
        if isinstance(cv, float):
            cv = f'{cv*100:.1f}%'
        name = m.replace(".sav", "")
        print(f'{name:25} | Acc: {str(acc):8} | CV: {cv}')
    except Exception as e:
        print(f'{m}: Error - {e}')
