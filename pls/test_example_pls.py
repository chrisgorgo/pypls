import pandas as pd
from pls import run_pls
import numpy as np

NEO_items = pd.read_csv("data/3760_NEO-FFI-3_20140312.csv", index_col=0, header=[0,1], na_values=["~<userSkipped>~", "DK"])
NEO_items = NEO_items.ix[:,2:62].dropna(axis=0)

ASR_items = pd.read_csv("data/3760_ASR_20140312.csv", index_col=0, header=[0,1], na_values=["~<userSkipped>~", "DK"])
columns = [column for column in ASR_items.columns if not column[0].strip().endswith("Describe:")]
ASR_items = ASR_items.ix[:, columns].ix[:,2:136].dropna(axis=0)

merged = NEO_items.join(ASR_items, how="inner")
neoffi = merged.ix[:,0:60]
print neoffi.shape

asr = merged.ix[:,60:194]
print asr.shape

neo_saliences, asr_saliences, salience_p_values, neo_saliences_bootstrap_ratios, asr_saliences_bootstrap_ratios = run_pls(np.array(neoffi), 
                                                                                                                          np.array(asr), 
                                                                                                                          n_components=40,
                                                                                                                          n_perm=500,
                                                                                                                          n_boot=500)

print "saliences_p_vals = %s"%str(list(salience_p_values))
print "neo_saliences_bootstrap_ratios.shape = %s"%str(neo_saliences_bootstrap_ratios.shape)
print "max(neo_saliences_bootstrap_ratios) = %s"%str(neo_saliences_bootstrap_ratios.max())
print "asr_saliences_bootstrap_ratios.shape = %s"%str(asr_saliences_bootstrap_ratios.shape)
print "max(asr_saliences_bootstrap_ratios) = %s"%str(asr_saliences_bootstrap_ratios.max())