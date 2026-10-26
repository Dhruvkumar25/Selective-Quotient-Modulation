# Ported from supervisor V8 notebook; see source_map.json.
print("Building shared independent validation rank-8 TDL-A basis bank ...")
t0 = time.time()
ind_tdla_bank = ind_build_tdla_basis_bank(rmax=max(IND_RANKS))
print(f"Bank shape: {ind_tdla_bank.shape}; built in {time.time()-t0:.1f} s")

