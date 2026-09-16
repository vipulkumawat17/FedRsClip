from __future__ import annotations

import json
import os
import pickle
import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {'.bmp','.gif','.jpeg','.jpg','.png','.tif','.tiff','.webp'}
MULTIBAND_DIR_NAMES = {'allbands','all_bands','all-band','all_bands_tif'}
DEFAULT_TEMPLATES = ['a photo of {}.','an image of {}.','a satellite photo of {}.']

@dataclass(frozen=True)
class PreparedDataset:
    name: str
    dataset_type: str
    work_dir: Path
    image_root: Path
    train_jsonl: Path
    val_jsonl: Path | None
    classnames: Path | None
    class_count: int
    train_count: int
    val_count: int

@dataclass(frozen=True)
class PrepareConfig:
    datasets_root: Path
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    force_extract: bool = False
    force_prepare: bool = False
    convert_multiband_tiff: bool = False
    rgb_bands: str = '4,3,2'
    download: bool = False

@dataclass(frozen=True)
class DatasetBuilder:
    canonical_name: str
    dataset_type: str
    build: Callable[[Path, bool], tuple[object, object, list[str]]]

DATASET_REGISTRY: dict[str, DatasetBuilder] = {}

# ---------- atomic write helpers ----------
def save_image_atomic(image: Image.Image, path: Path, **save_kwargs) -> None:
    """Save a PIL image without ever leaving a half-written file at `path`.

    Writes to a temp file then atomically renames it, so a process killed
    mid-write can never leave a truncated/corrupt file sitting at `path`
    that a later run's `if not path.exists(): ...` check would wrongly
    treat as "already done" forever.
    """
    if 'format' not in save_kwargs:
        ext = path.suffix.lower()
        format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.bmp': 'BMP', '.webp': 'WEBP'}
        save_kwargs['format'] = format_map.get(ext, 'PNG')
    tmp_path = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        image.save(tmp_path, **save_kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

# ---------- registry / optional TorchVision fallbacks ----------
def register_dataset(name, builder):
    key = normalize_dataset_key(name)
    DATASET_REGISTRY[key] = builder if isinstance(builder, DatasetBuilder) else DatasetBuilder(key,'TorchVision',builder)

def require_torchvision_datasets():
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError('torchvision is required for TorchVision fallback loading') from exc
    return datasets

def build_cifar10(root, download):
    d=require_torchvision_datasets(); tr=d.CIFAR10(root=root,train=True,download=download); va=d.CIFAR10(root=root,train=False,download=download)
    return tr,va,normalize_classnames(tr.classes)

def build_cifar100(root, download):
    d=require_torchvision_datasets(); tr=d.CIFAR100(root=root,train=True,download=download); va=d.CIFAR100(root=root,train=False,download=download)
    return tr,va,normalize_classnames(tr.classes)

def build_flowers102(root, download):
    d=require_torchvision_datasets(); tr=d.Flowers102(root=root,split='train',download=download); va=d.Flowers102(root=root,split='val',download=download)
    return tr,va,extract_class_names(tr,'flower class',102)

def build_food101(root, download):
    d=require_torchvision_datasets(); tr=d.Food101(root=root,split='train',download=download); va=d.Food101(root=root,split='test',download=download)
    return tr,va,normalize_classnames(tr.classes)

def build_caltech101(root, download):
    d=require_torchvision_datasets(); ds=d.Caltech101(root=root,download=download)
    return ds,ds,normalize_classnames(ds.categories)

def build_dataset(dataset_name, root, split='both', download=False):
    b=DATASET_REGISTRY.get(normalize_dataset_key(dataset_name))
    if b is None: raise ValueError(f"Unsupported dataset '{dataset_name}'")
    tr,va,classes=b.build(root,download)
    if split=='train': return tr,None,classes
    if split in {'val','validation','test'}: return None,va,classes
    return tr,va,classes

# ---------- main ----------
def prepare_dataset_source(source: Path, config: PrepareConfig, templates=None):
    templates=list(templates or DEFAULT_TEMPLATES)
    dataset_name,work_dir,raw_root=resolve_source(source,config)
    existing=prepared_from_existing_manifests(dataset_name,work_dir)
    if existing and not config.force_prepare and not is_multiband_path(existing.image_root): return existing
    b=DATASET_REGISTRY.get(normalize_dataset_key(dataset_name))
    if b is not None: return prepare_torchvision_dataset(dataset_name,b,work_dir,config,templates)
    if config.convert_multiband_tiff:
        converted=maybe_convert_multiband_tiff(raw_root,work_dir,parse_rgb_bands(config.rgb_bands))
        if converted is not None: raw_root=converted
    return prepare_imagefolder_dataset(dataset_name,work_dir,raw_root,config,templates)

def resolve_source(source: Path, config: PrepareConfig):
    if source.is_file() and source.suffix.lower()=='.zip':
        name=slugify(source.stem); work=config.datasets_root/name; raw=work/'raw'
        if config.force_extract or not raw.exists():
            raw.mkdir(parents=True,exist_ok=True); print(f'[EXTRACT] {source} -> {raw}'); safe_extract_zip(source,raw)
        return name,work,raw
    if source.is_dir():
        name=slugify(source.name); return name,source,source/'raw' if (source/'raw').exists() else source
    raise ValueError(f'Unsupported dataset source: {source}')

def prepare_torchvision_dataset(dataset_name,builder,work_dir,config,templates):
    local=prepare_local_registered_dataset(dataset_name,builder,work_dir,templates)
    if local is not None: return local
    try: return prepare_imagefolder_dataset(dataset_name,work_dir,work_dir/'raw',config,templates)
    except RuntimeError: pass
    tr,va,classes=build_dataset(dataset_name,work_dir,download=config.download)
    if tr is None or va is None: raise RuntimeError(f"Local preparation failed for '{dataset_name}', and TorchVision fallback returned no splits.")
    root=work_dir/'prepared_images'; tr_rows=export_indexed_dataset(tr,root,'train',classes,templates); va_rows=export_indexed_dataset(va,root,'val',classes,templates)
    write_prepared_files(work_dir,root,classes,tr_rows,va_rows)
    return PreparedDataset(slugify(builder.canonical_name),builder.dataset_type,work_dir,root,work_dir/'train.jsonl',work_dir/'val.jsonl',work_dir/'classnames.txt',len(classes),len(tr_rows),len(va_rows))

def prepare_local_registered_dataset(dataset_name,builder,work_dir,templates):
    key=normalize_dataset_key(dataset_name); raw=work_dir/'raw'
    if key=='cifar100': return prepare_local_cifar100(builder,work_dir,raw,templates)
    if key=='cifar10': return prepare_local_cifar10(builder,work_dir,raw,templates)
    if key=='flowers102': return prepare_local_flowers102(builder,work_dir,raw,templates)
    if key=='food101': return prepare_local_food101(builder,work_dir,raw,templates)
    if key=='caltech101': return prepare_local_caltech101(builder,work_dir,raw,templates)
    return None

# ---------- CIFAR ----------
def prepare_local_cifar100(builder,work_dir,raw,templates):
    root=first_existing_dir(raw/'cifar-100-python',raw); trf=root/'train'; tef=root/'test'; meta=root/'meta'
    if not all(p.exists() for p in (trf,tef,meta)): return None
    m=load_pickle(meta); classes=normalize_classnames(m.get('fine_label_names',[])); out=work_dir/'prepared_images'
    tr=export_cifar_pickle(trf,out,'train',classes,'fine_labels',templates); va=export_cifar_pickle(tef,out,'val',classes,'fine_labels',templates)
    write_prepared_files(work_dir,out,classes,tr,va); return prepared_from_rows(builder,work_dir,out,classes,tr,va)

def prepare_local_cifar10(builder,work_dir,raw,templates):
    root=first_existing_dir(raw/'cifar-10-batches-py',raw); batches=sorted(root.glob('data_batch_*')); tef=root/'test_batch'; meta=root/'batches.meta'
    if not batches or not tef.exists() or not meta.exists(): return None
    m=load_pickle(meta); classes=normalize_classnames(m.get('label_names',[])); out=work_dir/'prepared_images'; tr=[]
    for b in batches: tr.extend(export_cifar_pickle(b,out,'train',classes,'labels',templates,start_index=len(tr)))
    va=export_cifar_pickle(tef,out,'val',classes,'labels',templates); write_prepared_files(work_dir,out,classes,tr,va)
    return prepared_from_rows(builder,work_dir,out,classes,tr,va)

def export_cifar_pickle(path,out,split,classes,label_key,templates,start_index=0):
    import numpy as np
    payload=load_pickle(path); data=payload.get('data'); labels=payload.get(label_key)
    if data is None or labels is None: raise RuntimeError(f'{path} does not contain CIFAR data and {label_key}')
    data=np.asarray(data).reshape(-1,3,32,32).transpose(0,2,3,1); rng=random.Random(42); rows=[]
    for off,(arr,target) in enumerate(zip(data,labels)):
        cls=classes[int(target)]; d=out/safe_path_name(cls)/split
        d.mkdir(parents=True,exist_ok=True); p=out/split/safe_path_name(cls)/f'{start_index+off:08d}.png'; p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists(): save_image_atomic(Image.fromarray(arr.astype('uint8'),mode='RGB'), p)
        rows.append(row_for_image(p,out,cls,templates,rng))
    return rows

# ---------- Flowers-102 ----------
def prepare_local_flowers102(builder,work_dir,raw,templates):
    extract_nested_archives(raw,('*.tar','*.tar.gz','*.tgz'))
    tgz=first_existing_file(raw/'102flowers.tgz',raw/'jpg.tgz'); jpg=find_flowers_jpg_root(raw)
    # Flowers-102 always has 8189 total images. A `jpg/` dir that exists but
    # holds far fewer files means a previous extraction was interrupted --
    # `jpg.exists()` alone can't detect that.
    EXPECTED_FLOWERS102_IMAGES = 8189
    extracted_count = len(list_images(jpg)) if jpg.exists() else 0
    if tgz and extracted_count < EXPECTED_FLOWERS102_IMAGES:
        print(f'[EXTRACT] {tgz} -> {raw} (found {extracted_count}/{EXPECTED_FLOWERS102_IMAGES} images, re-extracting)')
        with tarfile.open(tgz) as a: safe_extract_tar(a,raw)
        jpg=find_flowers_jpg_root(raw)
    labels_path=find_file_by_name(raw,'imagelabels.mat'); setid=find_file_by_name(raw,'setid.mat')
    if not jpg.exists() or labels_path is None: return None
    labels,splits=read_flowers102_metadata(labels_path,setid); classes=[f'flower class {i}' for i in range(1,max(labels)+1)]
    tr_ids=splits.get('train') or list(range(1,int(len(labels)*.7)+1)); va_ids=splits.get('val') or list(range(int(len(labels)*.7)+1,len(labels)+1)); rng=random.Random(42)
    tr=rows_from_flowers_ids(tr_ids,labels,jpg,classes,templates,rng); va=rows_from_flowers_ids(va_ids,labels,jpg,classes,templates,rng)
    write_prepared_files(work_dir,jpg,classes,tr,va); return prepared_from_rows(builder,work_dir,jpg,classes,tr,va)

def read_flowers102_metadata(labels_path,setid_path):
    try: from scipy.io import loadmat
    except Exception as exc: raise RuntimeError('Flowers-102 preparation requires scipy') from exc
    m=loadmat(labels_path); labels=[int(x) for x in m['labels'].reshape(-1)]; s={}
    if setid_path and setid_path.exists():
        z=loadmat(setid_path); s['train']=flatten_mat_ints(z.get('trnid')); s['val']=flatten_mat_ints(z.get('valid')); s['test']=flatten_mat_ints(z.get('tstid'))
        if s['test']: s['val'].extend(s['test'])
    return labels,s

def rows_from_flowers_ids(ids,labels,jpg,classes,templates,rng):
    rows=[]
    for i in ids:
        if i<1 or i>len(labels): continue
        p=jpg/f'image_{i:05d}.jpg'
        if not p.exists(): continue
        idx=labels[i-1]-1
        if 0<=idx<len(classes): rows.append(row_for_image(p,jpg,classes[idx],templates,rng))
    return rows

# ---------- Food-101 ----------
def prepare_local_food101(builder,work_dir,raw,templates):
    extract_nested_archives(raw,('*.tar','*.tar.gz','*.tgz')); images=find_food101_images_root(raw); meta=find_food101_meta_root(raw)
    if images is None or meta is None: return None
    cf=first_existing_file(meta/'classes.txt',meta/'labels.txt'); classes=[display_class_name(x) for x in cf.read_text().splitlines() if x.strip()] if cf else [display_class_name(p.name) for p in sorted(images.iterdir()) if p.is_dir()]
    tr_items=read_food101_split(meta,'train'); va_items=read_food101_split(meta,'test')
    if not tr_items or not va_items: return None
    rng=random.Random(42); tr=rows_from_food101_items(tr_items,images,classes,templates,rng); va=rows_from_food101_items(va_items,images,classes,templates,rng)
    write_prepared_files(work_dir,images,classes,tr,va); return prepared_from_rows(builder,work_dir,images,classes,tr,va)

def read_food101_split(meta,split):
    jp=meta/f'{split}.json'; tx=meta/f'{split}.txt'
    if jp.exists():
        payload=json.loads(jp.read_text(encoding='utf-8')); out=[]
        if isinstance(payload,dict):
            for cls,names in payload.items():
                out.extend(str(n) if '/' in str(n) else f'{cls}/{n}' for n in names)
        elif isinstance(payload,list): out.extend(str(x) for x in payload)
        return out
    return [x.strip() for x in tx.read_text(encoding='utf-8').splitlines() if x.strip()] if tx.exists() else []

def rows_from_food101_items(items,images,classes,templates,rng):
    lookup={safe_path_name(x):x for x in classes}; rows=[]
    for item in items:
        rel=Path(item if item.endswith('.jpg') else f'{item}.jpg'); p=images/rel
        if not p.exists() or not rel.parts: continue
        cls=lookup.get(safe_path_name(rel.parts[0]),display_class_name(rel.parts[0])); rows.append(row_for_image(p,images,cls,templates,rng))
    return rows

# ---------- Caltech-101 ----------
def prepare_local_caltech101(builder,work_dir,raw,templates):
    archive=first_existing_file(raw/'caltech-101'/'101_ObjectCategories.tar.gz',raw/'Caltech-101'/'101_ObjectCategories.tar.gz',raw/'101_ObjectCategories.tar.gz')
    if archive:
        target=archive.parent/'101_ObjectCategories'
        if not target.exists():
            print(f'[CALTECH101] Extracting: {archive}')
            try:
                with tarfile.open(archive,'r:gz') as a: safe_extract_tar(a,archive.parent)
            except tarfile.ReadError as exc:
                # A bundled 101_ObjectCategories.tar.gz that's redundant/corrupt
                # shouldn't block using class folders that already exist on
                # disk (e.g. from a prior successful extraction, or a mirror
                # that ships pre-extracted folders alongside the archive).
                print(f'[WARNING] Could not extract {archive}: {exc}')
    root=find_caltech101_root(raw)
    if root is None:
        print(f'[CALTECH101] Could not find 101 class directories under {raw}'); return None
    excluded={'background_google','background-google','background google'}
    dirs=[p for p in sorted(root.iterdir()) if p.is_dir() and p.name.lower() not in excluded and list_images(p)]
    if len(dirs)<100:
        print(f'[CALTECH101] Only {len(dirs)} class directories found under {root}'); return None
    if len(dirs)>101: dirs=dirs[:101]
    print(f'[CALTECH101] Dataset root: {root}'); print(f'[CALTECH101] Classes found: {len(dirs)}')
    classes=[display_class_name(p.name) for p in dirs]; rng=random.Random(42); tr=[]; va=[]; te=[]
    for d,cls in zip(dirs,classes):
        imgs=list_images(d); rng.shuffle(imgs); a=int(len(imgs)*.70); b=a+int(len(imgs)*.15)
        if len(imgs)>=3: a=max(1,min(a,len(imgs)-2)); b=max(a+1,min(b,len(imgs)-1))
        tr.extend(make_rows(imgs[:a],root,cls,templates,rng)); va.extend(make_rows(imgs[a:b],root,cls,templates,rng)); te.extend(make_rows(imgs[b:],root,cls,templates,rng))
    write_prepared_files(work_dir,root,classes,tr,va,te); print(f'[CALTECH101] Prepared {len(tr)} train / {len(va)} validation / {len(te)} test images')
    return prepared_from_rows(builder,work_dir,root,classes,tr,va)

def find_caltech101_root(raw):
    candidates=[raw/'101_ObjectCategories',raw/'caltech-101'/'101_ObjectCategories',raw/'Caltech-101'/'101_ObjectCategories']
    for p in [*candidates,*[x for x in raw.rglob('101_ObjectCategories') if x.is_dir()]]:
        if not p.is_dir(): continue
        dirs=[d for d in p.iterdir() if d.is_dir() and d.name.lower() not in {'background_google','background-google'} and list_images(d)]
        if len(dirs)>=100: return p
    return None

# ---------- generic ImageFolder ----------
def prepare_imagefolder_dataset(dataset_name,work_dir,raw,config,templates):
    root=find_imagefolder_root(raw); dirs=sorted(p for p in root.iterdir() if p.is_dir() and list_images(p))
    if not dirs: raise RuntimeError(f'No class folders with images found under {root}')
    rng=random.Random(config.seed); tr=[]; va=[]; te=[]; classes=[display_class_name(p.name) for p in dirs]
    for d,cls in zip(dirs,classes):
        imgs=list_images(d); rng.shuffle(imgs); a=int(len(imgs)*config.train_ratio); b=a+int(len(imgs)*config.val_ratio)
        tr.extend(make_rows(imgs[:a],root,cls,templates,rng)); va.extend(make_rows(imgs[a:b],root,cls,templates,rng)); te.extend(make_rows(imgs[b:],root,cls,templates,rng))
    write_prepared_files(work_dir,root,classes,tr,va,te)
    return PreparedDataset(slugify(dataset_name),'ImageFolder',work_dir,root,work_dir/'train.jsonl',work_dir/'val.jsonl',work_dir/'classnames.txt',len(classes),len(tr),len(va))

def find_imagefolder_root(search_root):
    candidates=[]
    if not search_root.exists(): raise RuntimeError(f'Dataset root does not exist: {search_root}')
    for root in [search_root,*[p for p in search_root.rglob('*') if p.is_dir()]]:
        if is_multiband_path(root): continue
        dirs=[p for p in root.iterdir() if p.is_dir() and not is_multiband_path(p) and list_images(p)]
        if dirs: candidates.append((len(dirs),sum(len(list_images(d)) for d in dirs),root))
    if not candidates: raise RuntimeError(f'Could not find an ImageFolder-style class directory under {search_root}')
    candidates.sort(reverse=True); return candidates[0][2]

# ---------- manifests / archive utilities ----------
def prepared_from_rows(builder,work_dir,image_root,classes,tr,va):
    return PreparedDataset(slugify(builder.canonical_name),f'{builder.dataset_type}Local',work_dir,image_root,work_dir/'train.jsonl',work_dir/'val.jsonl',work_dir/'classnames.txt',len(classes),len(tr),len(va))

def manifest_sample_is_valid(jsonl_path: Path, image_root: Path, sample_size: int = 25) -> bool:
    """Spot-check that a cached manifest still matches what's on disk.

    Cached train.jsonl/val.jsonl are otherwise trusted indefinitely unless
    force_prepare is passed -- exactly how a stale manifest (from an
    interrupted run, or a raw dataset that got re-extracted/moved since)
    keeps getting silently reused forever. Sampling a handful of rows and
    checking the referenced image actually exists lets that be detected
    automatically and fall back to a full re-prepare.
    """
    lines: list[str] = []
    try:
        with jsonl_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
                if len(lines) >= 500:
                    break
    except OSError:
        return False
    if not lines:
        return False
    sample = random.Random(0).sample(lines, min(sample_size, len(lines)))
    hits = 0
    checked = 0
    for line in sample:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_value = item.get('image') or item.get('image_path') or item.get('path')
        if image_value is None:
            continue
        checked += 1
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if image_path.exists():
            hits += 1
    if checked == 0:
        return False
    return hits >= max(1, checked // 2)

def prepared_from_existing_manifests(dataset_name,work_dir):
    tr=work_dir/'train.jsonl'
    if not tr.exists(): return None
    va=work_dir/'val.jsonl'; cn=work_dir/'classnames.txt'; ir=work_dir/'image_root.txt'; root=Path(ir.read_text().strip()) if ir.exists() else infer_image_root(tr,work_dir); classes=read_classnames(cn) if cn.exists() else []
    if not manifest_sample_is_valid(tr, root):
        print(f'[STALE] {dataset_name}: cached manifest at {tr} does not match files on disk; re-preparing')
        return None
    typ='TorchVision' if DATASET_REGISTRY.get(normalize_dataset_key(dataset_name)) else 'ImageFolder'
    return PreparedDataset(slugify(dataset_name),typ,work_dir,root,tr,va if va.exists() else None,cn if cn.exists() else None,len(classes),count_jsonl_rows(tr),count_jsonl_rows(va) if va.exists() else 0)

def write_prepared_files(work_dir,image_root,classes,tr,va,te=None):
    write_jsonl(work_dir/'train.jsonl',tr); write_jsonl(work_dir/'val.jsonl',va)
    if te is not None: write_jsonl(work_dir/'test.jsonl',te)
    work_dir.mkdir(parents=True,exist_ok=True); (work_dir/'classnames.txt').write_text('\n'.join(classes)+'\n',encoding='utf-8'); (work_dir/'image_root.txt').write_text(str(image_root.resolve())+'\n',encoding='utf-8')

def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp_path = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp_path.open('w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=True)+'\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def make_rows(images,root,cls,templates,rng): return [row_for_image(p,root,cls,templates,rng) for p in images]

def row_for_image(p,root,cls,templates,rng): return {'image':p.relative_to(root).as_posix(),'caption':rng.choice(templates).format(cls),'label':cls,'class_name':cls}

def count_jsonl_rows(path):
    if not path.exists(): return 0
    with path.open('r',encoding='utf-8') as f: return sum(1 for x in f if x.strip())

def read_classnames(path): return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def infer_image_root(train_jsonl,work_dir):
    p=read_first_image_path(train_jsonl)
    if not p: return work_dir
    q=Path(p)
    if q.is_absolute() and q.exists(): return q.parent
    for d in [work_dir,*[x for x in work_dir.rglob('*') if x.is_dir()]]:
        if (d/q).exists(): return d
    return work_dir

def read_first_image_path(path):
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            for k in ('image','image_path','path'):
                if k in obj: return str(obj[k])
    return None

def safe_extract_zip(zip_path,output_dir):
    root=output_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target=(output_dir/m.filename).resolve()
            if root!=target and root not in target.parents: raise RuntimeError(f'Unsafe zip member path: {m.filename}')
        z.extractall(output_dir)

def extract_nested_archives(root,patterns):
    if not root.exists(): return
    for _ in range(5):
        found=False
        for pat in patterns:
            for p in sorted(root.rglob(pat)):
                # Skip macOS zip junk: AppleDouble resource-fork files
                # (named "._something") and anything under "__MACOSX/".
                # These match *.tar-style patterns by name but aren't real
                # archives and will error out of tarfile.
                if p.name.startswith('._') or '__MACOSX' in p.parts: continue
                marker=p.with_name('.'+p.name+'.extracted')
                if marker.exists(): continue
                print(f'[EXTRACT] {p} -> {p.parent}')
                try:
                    with tarfile.open(p) as a: safe_extract_tar(a,p.parent)
                except tarfile.ReadError as exc:
                    # Some mirrors bundle a redundant/corrupt archive next to
                    # data that's already extracted. Don't let it abort
                    # extraction of the others.
                    print(f'[WARNING] Could not extract {p}: {exc}')
                    marker.write_text('failed\n',encoding='utf-8'); continue
                marker.write_text('ok\n',encoding='utf-8'); found=True
        if not found: break

def safe_extract_tar(archive,output_dir):
    root=output_dir.resolve()
    for m in archive.getmembers():
        target=(output_dir/m.name).resolve()
        if root!=target and root not in target.parents: raise RuntimeError(f'Unsafe tar member path: {m.name}')
    archive.extractall(output_dir)

# ---------- discovery / TIFF ----------
def list_images(directory): return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if directory.exists() else []
def is_multiband_path(path): return path.name.lower() in MULTIBAND_DIR_NAMES
def is_probable_multiband_dataset(root):
    if not root.exists(): return False
    for p in [root,*[x for x in root.rglob('*') if x.is_dir()]]:
        if is_multiband_path(p) and any(c.is_dir() and list_images(c) for c in p.iterdir()): return True
    return False
def find_multiband_root(raw):
    if not raw.exists(): return None
    for p in [raw,*[x for x in raw.rglob('*') if x.is_dir()]]:
        if is_multiband_path(p) and any(c.is_dir() and list_images(c) for c in p.iterdir()): return p
    return None

def parse_rgb_bands(value):
    parts=[x.strip() for x in value.split(',') if x.strip()]
    if len(parts)!=3: raise ValueError('--rgb-bands must contain exactly three comma-separated 1-based band numbers')
    bands=tuple(int(x)-1 for x in parts)
    if any(x<0 for x in bands): raise ValueError('--rgb-bands values must be positive')
    return bands

def maybe_convert_multiband_tiff(raw,work,rgb_bands):
    root=find_multiband_root(raw)
    if root is None: return None
    out=work/'rgb_from_allbands'; marker=out/'.conversion_complete'
    if marker.exists(): return out
    try: import numpy as np; import tifffile
    except Exception as exc: raise RuntimeError('EuroSAT conversion requires numpy and tifffile') from exc
    for cls in [x for x in root.iterdir() if x.is_dir()]:
        od=out/cls.name; od.mkdir(parents=True,exist_ok=True)
        for p in sorted(cls.glob('*.tif'))+sorted(cls.glob('*.tiff')):
            q=od/f'{p.stem}.png'
            if q.exists(): continue
            save_image_atomic(Image.fromarray(multiband_to_rgb(tifffile.imread(str(p)),rgb_bands,np),mode='RGB'), q)
    marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text('ok\n'); return out

def multiband_to_rgb(array,bands,np):
    if array.ndim!=3: raise ValueError(f'Expected 3D TIFF, got {array.shape}')
    if array.shape[0]>=max(bands)+1 and array.shape[0]<=32: cf=array
    elif array.shape[-1]>=max(bands)+1 and array.shape[-1]<=32: cf=np.moveaxis(array,-1,0)
    else: raise ValueError(f'Could not identify band dimension for {array.shape}')
    rgb=np.stack([cf[i] for i in bands],axis=-1).astype(np.float32); out=np.zeros_like(rgb)
    for c in range(3):
        v=rgb[...,c]; lo,hi=np.percentile(v,(2,98)); hi=float(v.max()) if hi<=lo and v.max()>lo else (lo+1.0 if hi<=lo else hi); out[...,c]=np.clip((v-lo)/(hi-lo),0,1)
    return (out*255).round().astype(np.uint8)

# ---------- small helpers ----------
def load_pickle(path):
    with path.open('rb') as f: return pickle.load(f,encoding='latin1')
def first_existing_dir(*paths): return next((p for p in paths if p.exists() and p.is_dir()),paths[0])
def first_existing_file(*paths): return next((p for p in paths if p.exists() and p.is_file()),None)
def find_file_by_name(root,name):
    if not root.exists(): return None
    p=root/name
    if p.is_file(): return p
    return next((x for x in root.rglob(name) if x.is_file()),None)
def find_food101_images_root(raw):
    candidates=[raw/'images',raw/'food-101'/'images',*[x for x in raw.rglob('images') if x.is_dir()]]
    for p in candidates:
        if p.exists() and any(x.is_dir() and list_images(x) for x in p.iterdir()): return p
    return None
def find_food101_meta_root(raw):
    candidates=[raw/'meta',raw/'meta'/'meta',raw/'food-101'/'meta',*[x for x in raw.rglob('meta') if x.is_dir()]]
    for p in candidates:
        if (p/'train.json').exists() or (p/'train.txt').exists(): return p
    return None
def find_flowers_jpg_root(raw):
    candidates=[raw/'jpg',raw/'102flowers'/'jpg',*[x for x in raw.rglob('jpg') if x.is_dir()]]
    return next((p for p in candidates if p.exists() and list_images(p)),candidates[0])
def flatten_mat_ints(value): return [] if value is None else [int(x) for x in value.reshape(-1)]
def normalize_classnames(raw): return [display_class_name(x[0] if isinstance(x,(tuple,list)) else str(x)) for x in raw]
def extract_class_names(dataset,fallback_prefix,count):
    classes=getattr(dataset,'classes',None); return normalize_classnames(classes) if classes else [f'{fallback_prefix} {i}' for i in range(1,count+1)]
def display_class_name(value): return ' '.join(value.replace('_',' ').replace('-',' ').strip().split())
def safe_path_name(value):
    out=[]
    for c in value.lower():
        if c.isalnum(): out.append(c)
        elif out and out[-1]!='_': out.append('_')
    return ''.join(out).strip('_') or 'class'
def slugify(value):
    out=[]
    for c in value.lower():
        if c.isalnum(): out.append(c)
        elif out and out[-1]!='_': out.append('_')
    return ''.join(out).strip('_') or 'dataset'
def normalize_dataset_key(value): return ''.join(c for c in value.lower() if c.isalnum())

# ---------- registry: STL-10 intentionally removed; Caltech-101 replaces it ----------
register_dataset('cifar10',DatasetBuilder('cifar_10','TorchVision',build_cifar10)); register_dataset('cifar-10',DATASET_REGISTRY['cifar10']); register_dataset('cifar_10',DATASET_REGISTRY['cifar10'])
register_dataset('cifar100',DatasetBuilder('cifar_100','TorchVision',build_cifar100)); register_dataset('cifar-100',DATASET_REGISTRY['cifar100']); register_dataset('cifar_100',DATASET_REGISTRY['cifar100'])
register_dataset('flowers102',DatasetBuilder('flowers102','TorchVision',build_flowers102)); register_dataset('flower-102',DATASET_REGISTRY['flowers102']); register_dataset('flower_102',DATASET_REGISTRY['flowers102']); register_dataset('flower102',DATASET_REGISTRY['flowers102'])
register_dataset('food101',DatasetBuilder('food101','TorchVision',build_food101)); register_dataset('food-101',DATASET_REGISTRY['food101']); register_dataset('food_101',DATASET_REGISTRY['food101'])
register_dataset('caltech101',DatasetBuilder('caltech101','TorchVision',build_caltech101)); register_dataset('caltech-101',DATASET_REGISTRY['caltech101']); register_dataset('caltech_101',DATASET_REGISTRY['caltech101'])