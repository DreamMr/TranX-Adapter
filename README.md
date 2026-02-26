# *TranX-Adapter*

## 👀 TL;DR

**The Problem:** Current AI image detectors try to combine *artifact* with *semantic* features. However, artifact features often look very similar to one another, the MLLM gets "diluted" causing it to struggle with combining these two features effectively.

**The Solution:** ***TranX-Adapter***

**How It Works:**

- **Task-aware Optimal-Transport Fusion**: Uses a cost matrix (based on JS divergence) to smarter transfer artifact features into semantic features.
- **X-Fusion:** Uses cross-attention to transfer semantic features back into artifact features.

**The Result:** By forcing a better mixture of these two features, our method improves detection accuracy by up to 6% on standard AIGI benchmarks.



## 👨🏻‍💻Code

*Coming Soon*.
