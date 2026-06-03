import scanpy as sc


def _preprocess_and_hvg(adata, n_top_genes=3000, flavor="seurat_v3"):
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=3)

    adata_raw = adata.copy()
    sc.pp.highly_variable_genes(
        adata_raw,
        flavor=flavor,
        n_top_genes=min(n_top_genes, adata_raw.n_vars),
        subset=False,
        inplace=True,
    )

    adata.var["highly_variable"] = adata_raw.var["highly_variable"]
    for col in [
        "highly_variable_rank",
        "means",
        "variances",
        "variances_norm",
        "highly_variable_nbatches",
    ]:
        if col in adata_raw.var.columns:
            adata.var[col] = adata_raw.var[col]

    del adata_raw

    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)

    return adata


def _gene_intersection(adatas, use_hvg=True, reference=0):
    if use_hvg:
        sets = [set(ad.var_names[ad.var["highly_variable"].values]) for ad in adatas]
    else:
        sets = [set(ad.var_names) for ad in adatas]

    common = set.intersection(*sets)
    if not common:
        raise ValueError("No common genes across samples after HVG filtering.")

    ref = adatas[reference].var_names
    genes = [g for g in ref if g in common]
    return genes


def load_and_preprocess_slices(
    paths,
    section_ids=None,
    n_top_genes=7500,
    flavor="seurat_v3",
    domain_key=None,
    data_dir="",
):
    final_section_ids = section_ids if section_ids is not None else paths
    adatas = []

    for p in paths:
        path = f"{data_dir}{p}" if data_dir else p
        ad = sc.read_h5ad(path)
        ad.var_names_make_unique()
        ad.obs_names_make_unique()

        ad = _preprocess_and_hvg(
            adata=ad,
            n_top_genes=n_top_genes,
            flavor=flavor,
        )

        if domain_key is not None and domain_key in ad.obs.columns:
            ad = ad[~ad.obs[domain_key].isna(), :].copy()

        adatas.append(ad)

    print(f"[GRASS] loaded {len(adatas)} slices")
    return adatas, final_section_ids


def align_genes_across_slices(
    adatas,
    use_hvg=True,
    reference=0,
):
    genes = _gene_intersection(
        adatas=adatas,
        use_hvg=use_hvg,
        reference=reference,
    )

    aligned = []
    for ad in adatas:
        ad_sub = ad[:, genes].copy()
        ad_sub.var_names_make_unique()
        ad_sub.obs_names_make_unique()
        aligned.append(ad_sub)

    print(f"[GRASS] aligned genes: {len(genes)}")
    return aligned, genes


def annotate_section_ids(adatas, section_ids, key="section"):
    if len(adatas) != len(section_ids):
        raise ValueError("adatas and section_ids must have the same length")

    for ad, sid in zip(adatas, section_ids):
        ad.obs[key] = str(sid)

    return adatas


def print_slice_shapes(adatas):
    shapes = [ad.shape for ad in adatas]
    print(f"[GRASS] slice shapes: {shapes}")